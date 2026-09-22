import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { openRepository } from "./online.mjs";

const call = { tool: "bash", arguments: { command: "npm run test" } };
const completed = { durationMs: 100, status: "completed" };

async function workspace(t) {
	const directory = await mkdtemp(join(tmpdir(), "volt-online-"));
	t.after(async () => {
		assert.equal(dirname(directory), resolve(tmpdir()));
		assert.ok(directory.startsWith(join(tmpdir(), "volt-online-")));
		await rm(directory, { recursive: true, force: true });
	});
	const repo = join(directory, "repo");
	const stateDirectory = join(directory, "state");
	await mkdir(join(repo, ".git"), { recursive: true });
	return { directory, repo, stateDirectory };
}

test("scores captured predictions before updating, persists learning, and stores no command text", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const baseBefore = await readFile(new URL("./model.json", import.meta.url), "utf8");
	const learner = await openRepository(repo, { stateDirectory });
	const ticket = learner.begin(call);
	assert.equal(ticket.predicted.adaptive, "long");
	const result = await learner.complete(ticket, completed);
	assert.equal(result.before.prediction.longScore, ticket.prediction.longScore);
	assert.equal(result.actual, "short");
	assert.equal(learner.report().metrics.adaptive.shortAsLong, 1);
	assert.ok(learner.begin(call).prediction.longScore < ticket.prediction.longScore);
	const reopened = await openRepository(repo, { stateDirectory });
	assert.deepEqual(reopened.begin(call), learner.begin(call));
	assert.equal(reopened.begin(call).predicted.history, "short");
	const saved = await readFile(learner.statePath, "utf8");
	assert.ok(!saved.includes(call.arguments.command));
	assert.equal(await readFile(new URL("./model.json", import.meta.url), "utf8"), baseBefore);
});

test("repeated measured results adapt one repository while other repositories retain the shipped model", async (t) => {
	const { directory, repo, stateDirectory } = await workspace(t);
	const otherRepo = join(directory, "other");
	await mkdir(join(otherRepo, ".git"), { recursive: true });
	await mkdir(join(repo, "src"));
	const learner = await openRepository(join(repo, "src"), { stateDirectory });
	const root = await openRepository(repo, { stateDirectory });
	const other = await openRepository(otherRepo, { stateDirectory });
	assert.equal(learner.statePath, root.statePath);
	assert.notEqual(learner.statePath, other.statePath);
	for (let index = 0; index < 80; index++) await learner.complete(learner.begin(call), completed);
	assert.equal(learner.begin(call).prediction.durationClass, "short");
	assert.equal(other.begin(call).prediction.durationClass, "long");
	assert.ok(learner.report().metrics.adaptive.accuracyOnDecided > learner.report().metrics.shipped.accuracyOnDecided);
});

test("failed and cancelled calls do not train or enter successful prediction metrics", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const learner = await openRepository(repo, { stateDirectory });
	const before = learner.begin(call).prediction;
	await learner.complete(learner.begin(call), { durationMs: 100, status: "failed" });
	await learner.complete(learner.begin(call), { durationMs: 3000, status: "cancelled" });
	assert.deepEqual(learner.begin(call).prediction, before);
	assert.equal(learner.report().successful, 0);
	assert.equal(learner.report().failed, 1);
	assert.equal(learner.report().cancelled, 1);
	assert.equal(learner.report().metrics.adaptive.total, 0);
	assert.equal(learner.report().rememberedCommands, 0);
});

test("overlapping calls retain their start-time predictions and independent writers merge updates", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const first = await openRepository(repo, { stateDirectory });
	const second = await openRepository(repo, { stateDirectory });
	const tickets = [first.begin(call), first.begin(call), second.begin(call)];
	await Promise.all([
		first.complete(tickets[0], completed),
		first.complete(tickets[1], completed),
		second.complete(tickets[2], completed),
	]);
	await first.refresh();
	assert.equal(first.report().successful, 3);
	assert.equal(first.report().metrics.adaptive.shortAsLong, 3);
	assert.equal(first.report().metrics.history.uncertain, 3);
	await assert.rejects(first.complete(tickets[0], completed), /already consumed/);
});

test("validates observations and refuses corrupt state without replacing it", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const learner = await openRepository(repo, { stateDirectory });
	const ticket = learner.begin(call);
	await assert.rejects(learner.complete(ticket, { durationMs: NaN, status: "completed" }), /nonnegative/);
	await assert.rejects(learner.complete(ticket, { durationMs: 2, status: "timeout" }), /status/);
	await learner.complete(ticket, { durationMs: 2000, status: "completed" });
	assert.equal(learner.report().metrics.adaptive.shortAsLong, 1);
	await writeFile(learner.statePath, "corrupt");
	await assert.rejects(openRepository(repo, { stateDirectory }), SyntaxError);
	await assert.rejects(learner.complete(learner.begin(call), completed), SyntaxError);
	assert.equal(await readFile(learner.statePath, "utf8"), "corrupt");
	await assert.rejects(readFile(`${learner.statePath}.lock`), { code: "ENOENT" });
});

test("a busy writer is reported without deleting another process's lock", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const learner = await openRepository(repo, { stateDirectory });
	await mkdir(stateDirectory);
	await writeFile(`${learner.statePath}.lock`, "another owner");
	await assert.rejects(learner.complete(learner.begin(call), completed), /busy/);
	assert.equal(await readFile(`${learner.statePath}.lock`, "utf8"), "another owner");
	assert.equal(learner.report().successful, 0);
});

test("command history stays bounded", async (t) => {
	const { repo, stateDirectory } = await workspace(t);
	const learner = await openRepository(repo, { stateDirectory });
	for (let index = 0; index < 130; index++) {
		await learner.complete(learner.begin({ tool: "bash", arguments: { command: `echo ${index}` } }), completed);
	}
	assert.equal(learner.report().rememberedCommands, 128);
});

test("CLI can replay observations concurrently across processes without lost writes", async (t) => {
	const { directory, repo, stateDirectory } = await workspace(t);
	const input = join(directory, "observation.json");
	await writeFile(input, JSON.stringify({ ...call, ...completed }));
	const cli = fileURLToPath(new URL("./cli.mjs", import.meta.url));
	await Promise.all(Array.from({ length: 4 }, () => new Promise((resolvePromise, reject) => {
		const child = spawn(process.execPath, [cli, "observe", "--repo", repo, "--state-dir", stateDirectory, "--input", input]);
		let error = "";
		child.stdout.resume();
		child.stderr.on("data", (data) => { error += data; });
		child.on("error", reject);
		child.on("close", (code) => code === 0 ? resolvePromise() : reject(new Error(error)));
	})));
	const learner = await openRepository(repo, { stateDirectory });
	assert.equal(learner.report().successful, 4);
	assert.equal(learner.report().metrics.adaptive.total, 4);
});
