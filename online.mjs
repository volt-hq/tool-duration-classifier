import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile, realpath, rename, stat, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { setTimeout } from "node:timers/promises";
import { baseline, features, predict, validateModel } from "./model.mjs";

const PREDICTORS = ["adaptive", "adaptiveWithAbstention", "shipped", "rules", "history"];
const HISTORY_LIMIT = 128;
const LEARNING_RATE = 0.1;
const ANCHOR_STRENGTH = 0.001;

export async function repositoryRoot(directory) {
	const original = await realpath(resolve(directory));
	if (!(await stat(original)).isDirectory()) throw new Error("Repository path must be a directory");
	let current = original;
	while (true) {
		try { await stat(join(current, ".git")); return current; }
		catch (error) { if (error.code !== "ENOENT") throw error; }
		const parent = dirname(current);
		if (parent === current) return original;
		current = parent;
	}
}

function countPrediction(counts, predicted, actual) {
	if (predicted === "uncertain") { counts.uncertain++; return; }
	counts[`${actual}As${predicted === "long" ? "Long" : "Short"}`]++;
}

function summarize(counts) {
	const decided = counts.longAsLong + counts.longAsShort + counts.shortAsLong + counts.shortAsShort;
	const total = decided + counts.uncertain;
	return { ...counts, total, coverage: total ? decided / total : null,
		accuracyOnDecided: decided ? (counts.longAsLong + counts.shortAsShort) / decided : null };
}

function validateState(state, baseHash, repository) {
	if (!state || state.repository !== repository || state.baseHash !== baseHash) {
		throw new Error("Local state belongs to another repository or shipped model; use a separate state directory");
	}
	validateModel(state.model);
	if (!Number.isSafeInteger(state.successful) || state.successful < 0 ||
		!Number.isSafeInteger(state.failed) || state.failed < 0 ||
		!Number.isSafeInteger(state.cancelled) || state.cancelled < 0 ||
		!Array.isArray(state.history) || state.history.length > HISTORY_LIMIT ||
		!state.history.every((entry) => /^[a-f0-9]{64}$/.test(entry.key) &&
			Number.isSafeInteger(entry.count) && entry.count > 0 && Number.isFinite(entry.meanMs) && entry.meanMs >= 0) ||
		!PREDICTORS.every((name) => state.metrics?.[name] &&
			["longAsLong", "longAsShort", "shortAsLong", "shortAsShort", "uncertain"].every((key) =>
				Number.isSafeInteger(state.metrics[name][key]) && state.metrics[name][key] >= 0) &&
			summarize(state.metrics[name]).total === state.successful)) {
		throw new Error("Invalid local learning state");
	}
	return state;
}

/** Updates only after a completed observation. Regularization anchors weights to the shipped model. */
function updateModel(model, base, vector, tool, actual) {
	let logit = model.bias;
	for (const [slot, value] of vector) logit += model.weights[slot] * value;
	const error = 1 / (1 + Math.exp(-logit)) - Number(actual === "long");
	model.bias -= LEARNING_RATE * (error + ANCHOR_STRENGTH * (model.bias - base.bias));
	for (let index = 0; index < model.weights.length; index++) {
		model.weights[index] -= LEARNING_RATE * ANCHOR_STRENGTH * (model.weights[index] - base.weights[index]);
	}
	for (const [slot, value] of vector) model.weights[slot] -= LEARNING_RATE * error * value;
	if (!model.tools.includes(tool)) model.tools.push(tool);
	model.training.basis = base.training.basis === "measured" ? "measured" : "mixed";
}

/** Local, per-working-tree state. No command arguments, outputs, or raw observations are persisted. */
export async function openRepository(directory, options = {}) {
	let repository = await repositoryRoot(directory);
	if (process.platform === "win32") repository = repository.toLowerCase();
	const baseBytes = await readFile(options.modelPath ?? new URL("./model.json", import.meta.url), "utf8");
	const base = validateModel(JSON.parse(baseBytes));
	const baseHash = createHash("sha256").update(baseBytes).digest("hex");
	const key = createHash("sha256").update(repository).digest("hex");
	const stateDirectory = resolve(options.stateDirectory ?? process.env.VOLT_DURATION_STATE_DIR ?? join(homedir(), ".volt", "tool-duration-classifier"));
	const statePath = join(stateDirectory, `${key}.json`);
	const lockPath = `${statePath}.lock`;
	const fresh = () => ({
		repository, baseHash, model: structuredClone(base), successful: 0, failed: 0, cancelled: 0, history: [],
		metrics: Object.fromEntries(PREDICTORS.map((name) => [name,
			{ longAsLong: 0, longAsShort: 0, shortAsLong: 0, shortAsShort: 0, uncertain: 0 }])),
	});
	async function load() {
		try {
			const saved = validateState(JSON.parse(await readFile(statePath, "utf8")), baseHash, repository);
			if (saved.model.thresholdMs !== base.thresholdMs) throw new Error("Local duration threshold differs from shipped model");
			return saved;
		} catch (error) { if (error.code === "ENOENT") return fresh(); throw error; }
	}
	let state = await load();
	let pending = Promise.resolve();
	const tickets = new WeakMap();
	return {
		statePath,
		begin(call) {
			const vector = features(call);
			const commandKey = createHash("sha256").update(JSON.stringify([call.tool,
				Object.fromEntries(Object.entries(call.arguments).sort(([a], [b]) => a.localeCompare(b)))])).digest("hex");
			const history = state.history.find((entry) => entry.key === commandKey);
			const prediction = Object.freeze(predict(state.model, call));
			const predicted = Object.freeze({
				adaptive: prediction.longScore >= 0.5 ? "long" : "short",
				adaptiveWithAbstention: prediction.durationClass,
				shipped: predict(base, call).longScore >= 0.5 ? "long" : "short",
				rules: baseline(call),
				history: history ? (history.meanMs > base.thresholdMs ? "long" : "short") : "uncertain",
			});
			const ticket = Object.freeze({ prediction, predicted, learnedObservations: state.successful });
			tickets.set(ticket, { vector, commandKey, tool: call.tool });
			return ticket;
		},
		complete(ticket, { durationMs, status }) {
			if (!tickets.has(ticket)) return Promise.reject(new Error("Unknown or already consumed observation"));
			if (!Number.isFinite(durationMs) || durationMs < 0 || !["completed", "failed", "cancelled"].includes(status)) {
				return Promise.reject(new Error("Observation requires nonnegative durationMs and completed/failed/cancelled status"));
			}
			const captured = tickets.get(ticket);
			tickets.delete(ticket);
			const operation = pending.then(async () => {
				await mkdir(stateDirectory, { recursive: true, mode: 0o700 });
				let lock;
				for (let attempt = 0; attempt < 26; attempt++) {
					try { lock = await open(lockPath, "wx", 0o600); break; }
					catch (error) {
						if (error.code !== "EEXIST") throw error;
						if (attempt === 25) throw new Error(`Local learner is busy: ${lockPath}. If its owner crashed, remove this lock after all observers stop.`);
						await setTimeout(20);
					}
				}
				const temporary = `${statePath}.${randomUUID()}.tmp`;
				try {
					await lock.writeFile(String(process.pid));
					const next = await load();
					let actual = null;
					if (status === "completed") {
						actual = durationMs > base.thresholdMs ? "long" : "short";
						// Score the predictions captured at start, even if other calls finished in the meantime.
						for (const name of PREDICTORS) countPrediction(next.metrics[name], ticket.predicted[name], actual);
						updateModel(next.model, base, captured.vector, captured.tool, actual);
						next.successful++;
						const previous = next.history.find((entry) => entry.key === captured.commandKey);
						next.history = next.history.filter((entry) => entry.key !== captured.commandKey);
						next.history.push({ key: captured.commandKey, count: (previous?.count ?? 0) + 1,
							meanMs: previous ? previous.meanMs * 0.8 + durationMs * 0.2 : durationMs });
						if (next.history.length > HISTORY_LIMIT) next.history.shift();
					} else next[status]++;
					next.updatedAt = new Date().toISOString();
					await writeFile(temporary, `${JSON.stringify(next)}\n`, { flag: "wx", mode: 0o600 });
					await rename(temporary, statePath);
					state = next;
					return { before: ticket, actual, status, durationMs, successful: state.successful };
				} finally {
					try { await unlink(temporary).catch((error) => { if (error.code !== "ENOENT") throw error; }); }
					finally {
						try { await lock.close(); }
						finally { await unlink(lockPath); }
					}
				}
			});
			pending = operation.catch(() => {});
			return operation;
		},
		async refresh() { await pending; state = await load(); },
		async flush() { await pending; },
		report() {
			return {
				repository, statePath, thresholdMs: base.thresholdMs, successful: state.successful,
				failed: state.failed, cancelled: state.cancelled, rememberedCommands: state.history.length,
				metrics: Object.fromEntries(PREDICTORS.map((name) => [name, summarize(state.metrics[name])])),
			};
		},
	};
}
