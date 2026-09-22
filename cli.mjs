#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";
import { evaluate, predict, train, validateDataset, validateModel } from "./model.mjs";
import { openRepository } from "./online.mjs";

const HELP = `Local duration classifier experiment (does not execute tool calls).

node cli.mjs train [--data dataset.json] [--model model.json]
node cli.mjs evaluate [--data dataset.json] [--model model.json]
node cli.mjs predict --command "npm run test" [--model model.json]
node cli.mjs predict --input call.json [--model model.json]
node cli.mjs benchmark [--data dataset.json] [--model model.json]
node cli.mjs predict --repo <directory> --command "npm run test" [--state-dir <directory>]
node cli.mjs observe --repo <directory> --input observation.json [--state-dir <directory>]
node cli.mjs status --repo <directory> [--state-dir <directory>]

Defaults use seed-data.json and model.json beside this script. Training writes the model.
--input reads { "tool": "bash", "arguments": { "command": "npm run test" } }.
Scores from synthetic seed labels are not calibrated real-world probabilities.
observe learns from a supplied completed observation; it never executes the command.
An observation is a call plus durationMs and status (completed, failed, or cancelled).
Repository models stay under ~/.volt/tool-duration-classifier by default.
`;

async function readCall(options) {
	if (Number(Boolean(options["--command"])) + Number(Boolean(options["--input"])) !== 1) {
		throw new Error("Predict requires exactly one of --command or --input");
	}
	return options["--input"] ? JSON.parse(await readFile(options["--input"], "utf8")) :
		{ tool: "bash", arguments: { command: options["--command"] } };
}

export async function main(args) {
	if (args.length === 0 || args.includes("--help")) { console.log(HELP); return; }
	const [action, ...flags] = args;
	if (!["train", "evaluate", "predict", "benchmark", "observe", "status"].includes(action)) throw new Error(`Unknown action: ${action}`);
	const allowed = action === "predict" ? ["--model", "--command", "--input", "--repo", "--state-dir"] :
		action === "observe" ? ["--model", "--input", "--repo", "--state-dir"] :
			action === "status" ? ["--model", "--repo", "--state-dir"] : ["--data", "--model"];
	const options = {};
	for (let index = 0; index < flags.length; index += 2) {
		const flag = flags[index];
		if (!allowed.includes(flag) || options[flag] !== undefined || !flags[index + 1] || flags[index + 1].startsWith("--")) {
			throw new Error(`Invalid, duplicate, or missing option: ${flag}`);
		}
		options[flag] = flags[index + 1];
	}
	const modelPath = options["--model"] ?? fileURLToPath(new URL("./model.json", import.meta.url));
	const dataPath = options["--data"] ?? fileURLToPath(new URL("./seed-data.json", import.meta.url));
	if (action === "observe" || action === "status" || options["--repo"] || options["--state-dir"]) {
		if (!options["--repo"]) throw new Error("Local learning requires --repo <directory>");
		const learner = await openRepository(options["--repo"], { modelPath, stateDirectory: options["--state-dir"] });
		if (action === "status") { console.log(JSON.stringify(learner.report(), null, 2)); return; }
		if (action === "predict") { console.log(JSON.stringify(learner.begin(await readCall(options)), null, 2)); return; }
		if (!options["--input"]) throw new Error("Observe requires --input observation.json");
		const observation = JSON.parse(await readFile(options["--input"], "utf8"));
		const result = await learner.complete(learner.begin(observation), observation);
		console.log(JSON.stringify({ ...result, report: learner.report() }, null, 2));
		return;
	}
	if (action === "train") {
		const model = train(JSON.parse(await readFile(dataPath, "utf8")));
		const serialized = `${JSON.stringify(model)}\n`;
		await writeFile(modelPath, serialized);
		console.log(JSON.stringify({ modelPath, bytes: Buffer.byteLength(serialized), training: model.training }, null, 2));
		return;
	}
	const loadStart = performance.now();
	const serialized = await readFile(modelPath, "utf8");
	const model = validateModel(JSON.parse(serialized));
	const loadMs = performance.now() - loadStart;
	if (action === "predict") {
		console.log(JSON.stringify(predict(model, await readCall(options)), null, 2));
		return;
	}
	const dataset = JSON.parse(await readFile(dataPath, "utf8"));
	if (action === "evaluate") {
		console.log(JSON.stringify(evaluate(model, dataset), null, 2));
		return;
	}
	const calls = validateDataset(dataset).examples.filter((example) => example.split === "test");
	const iterations = 10000;
	for (let index = 0; index < 1000; index++) predict(model, calls[index % calls.length]);
	const elapsed = [];
	let checksum = 0;
	for (let index = 0; index < iterations; index++) {
		const start = performance.now();
		checksum += predict(model, calls[index % calls.length]).longScore;
		elapsed.push(performance.now() - start);
	}
	elapsed.sort((a, b) => a - b);
	console.log(JSON.stringify({
		node: process.version, platform: process.platform, arch: process.arch,
		modelBytes: Buffer.byteLength(serialized), loadMs, iterations,
		meanMs: elapsed.reduce((sum, ms) => sum + ms, 0) / iterations,
		p50Ms: elapsed[Math.floor(iterations * 0.5)], p95Ms: elapsed[Math.floor(iterations * 0.95)], checksum,
		note: "Warm in-process inference including feature extraction; load excludes Node startup. Seed-sized inputs only.",
	}, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main(process.argv.slice(2)).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
