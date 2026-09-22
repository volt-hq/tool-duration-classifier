import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { evaluate, features, predict, train, validateDataset, validateModel } from "./model.mjs";

const dataset = JSON.parse(await readFile(new URL("./seed-data.json", import.meta.url), "utf8"));
const model = validateModel(JSON.parse(await readFile(new URL("./model.json", import.meta.url), "utf8")));
const call = (command) => ({ tool: "bash", arguments: { command } });

test("checked-in model reproduces deterministic training and survives JSON roundtrip", () => {
	const trained = train(dataset);
	assert.deepEqual(trained, model);
	const loaded = validateModel(JSON.parse(JSON.stringify(trained)));
	assert.deepEqual(predict(loaded, call("npm run test")), predict(trained, call("npm run test")));
	assert.equal(predict(loaded, call("npm run test")).durationClass, "long");
	assert.equal(predict(loaded, call("git status --short")).durationClass, "short");
});

test("changing holdout labels cannot affect learned parameters", () => {
	const changed = structuredClone(dataset);
	for (const example of changed.examples) {
		if (example.split === "test") example.label = example.label === "long" ? "short" : "long";
	}
	assert.deepEqual(train(changed), model);
});

test("trainer learns supplied labels, rather than encoding the command keyword baseline", () => {
	const reversed = structuredClone(dataset);
	for (const example of reversed.examples) example.label = example.label === "long" ? "short" : "long";
	const trained = train(reversed);
	assert.equal(predict(trained, call("npm run test")).durationClass, "short");
	assert.equal(predict(trained, call("git status --short")).durationClass, "long");
});

test("measured durations determine labels using a strict greater-than boundary", () => {
	const measured = structuredClone(dataset);
	for (const example of measured.examples) {
		example.durationMs = example.label === "long" ? 2001 : 2000;
		delete example.label;
	}
	const checked = validateDataset(measured);
	assert.deepEqual(checked.examples.map((example) => example.label), dataset.examples.map((example) => example.label));
	assert.equal(train(measured).training.basis, "measured");
	measured.examples[0].durationMs = -1;
	assert.throws(() => validateDataset(measured), /nonnegative/);
});

test("rejects group leakage and identical features across train/test", () => {
	const leakedGroup = structuredClone(dataset);
	leakedGroup.examples.find((example) => example.split === "test").group = dataset.examples[0].group;
	assert.throws(() => validateDataset(leakedGroup), /leaks across/);
	const duplicate = structuredClone(dataset);
	duplicate.examples.push({ ...duplicate.examples[0], group: "duplicate", split: "test" });
	assert.throws(() => validateDataset(duplicate), /duplicate call/);
});

test("rejects invalid models, calls, datasets, and mismatched evaluation thresholds", () => {
	assert.throws(() => validateModel({ ...model, weights: [0] }), /Invalid duration model/);
	assert.throws(() => validateModel({ ...model, bias: Infinity }), /Invalid duration model/);
	assert.throws(() => predict(model, { tool: "bash", arguments: "npm test" }), /Expected/);
	assert.throws(() => validateDataset({ ...dataset, thresholdMs: 0 }), /thresholdMs/);
	const ambiguous = structuredClone(dataset);
	ambiguous.examples[0].durationMs = 3000;
	assert.throws(() => validateDataset(ambiguous), /either/);
	assert.throws(() => evaluate(model, { ...dataset, thresholdMs: 5000 }), /thresholds differ/);
});

test("unseen tools abstain even when their argument looks like a familiar slow command", () => {
	const result = predict(model, { tool: "unseen_remote_tool", arguments: { command: "npm run test" } });
	assert.equal(result.durationClass, "uncertain");
	assert.equal(result.reason, "unseen tool");
});

test("feature extraction ignores argument key order", () => {
	assert.deepEqual(features({ tool: "read", arguments: { path: "file.ts", limit: 100 } }),
		features({ tool: "read", arguments: { limit: 100, path: "file.ts" } }));
});

test("evaluation accounts for every holdout and reports uncertainty separately", () => {
	const report = evaluate(model, dataset);
	assert.equal(report.testBasis, "synthetic");
	for (const name of ["modelForcedAtHalf", "modelWithAbstention", "keywordBaseline", "alwaysShortBaseline"]) {
		const entry = report[name];
		assert.equal(entry.longAsLong + entry.longAsShort + entry.shortAsLong + entry.shortAsShort + entry.uncertain,
			report.testExamples);
	}
	assert.equal(report.errorsAtHalf.length, report.modelForcedAtHalf.longAsShort + report.modelForcedAtHalf.shortAsLong);
});

test("CLI trains and predicts offline without executing input commands", async () => {
	const directory = await mkdtemp(join(tmpdir(), "volt-duration-test-"));
	try {
		const cli = fileURLToPath(new URL("./cli.mjs", import.meta.url));
		const savedModel = join(directory, "model.json");
		const trained = spawnSync(process.execPath, [cli, "train", "--model", savedModel], { encoding: "utf8" });
		assert.equal(trained.status, 0, trained.stderr);
		const marker = join(directory, "must-not-exist.txt");
		const input = join(directory, "call.json");
		const script = `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'executed')`;
		await writeFile(input, JSON.stringify(call(`node -e ${JSON.stringify(script)}`)));
		const result = spawnSync(process.execPath, [cli, "predict", "--model", savedModel, "--input", input], { encoding: "utf8" });
		assert.equal(result.status, 0, result.stderr);
		assert.equal(JSON.parse(result.stdout).thresholdMs, 2000);
		await assert.rejects(readFile(marker), { code: "ENOENT" });
		const bad = spawnSync(process.execPath, [cli, "predict", "--input", input, "--command", "pwd"], { encoding: "utf8" });
		assert.notEqual(bad.status, 0);
		assert.match(bad.stderr, /exactly one/);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});
