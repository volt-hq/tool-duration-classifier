// Standalone experiment: predicts duration, never permission or scheduling safety.
export const DIMENSIONS = 2048;
export const MAX_INPUT_CHARACTERS = 4096;

export function validateCall(call) {
	if (!call || typeof call.tool !== "string" || !/^[a-zA-Z0-9_.:-]{1,64}$/.test(call.tool) ||
		!call.arguments || typeof call.arguments !== "object" || Array.isArray(call.arguments)) {
		throw new Error("Expected { tool: nonempty name, arguments: object }");
	}
}

export function features(call) {
	validateCall(call);
	// Sorting top-level keys makes ordinary tool argument order irrelevant.
	const args = Object.fromEntries(Object.entries(call.arguments).sort(([a], [b]) => a.localeCompare(b)));
	const text = JSON.stringify(args).slice(0, MAX_INPUT_CHARACTERS).toLowerCase();
	const tokens = text.match(/[a-z0-9_./:-]+|&&|\|\||[;|]/g) ?? [];
	const names = new Set([`tool:${call.tool}`]);
	for (let index = 0; index < tokens.length; index++) {
		names.add(`word:${tokens[index]}`);
		if (index > 0) names.add(`pair:${tokens[index - 1]} ${tokens[index]}`);
	}
	const vector = new Map();
	for (const name of names) {
		let hash = 2166136261;
		for (let index = 0; index < name.length; index++) hash = Math.imul(hash ^ name.charCodeAt(index), 16777619);
		const slot = (hash >>> 0) % DIMENSIONS;
		vector.set(slot, (vector.get(slot) ?? 0) + (hash < 0 ? -1 : 1));
	}
	const norm = Math.sqrt([...vector.values()].reduce((sum, value) => sum + value * value, 0)) || 1;
	return [...vector].map(([slot, value]) => [slot, value / norm]);
}

function scoreVector(model, vector) {
	let logit = model.bias;
	for (const [slot, value] of vector) logit += model.weights[slot] * value;
	return 1 / (1 + Math.exp(-logit));
}

export function validateModel(model) {
	if (!model || model.algorithm !== "hashed-logistic-regression" || model.dimensions !== DIMENSIONS ||
		model.maxInputCharacters !== MAX_INPUT_CHARACTERS ||
		!Number.isFinite(model.thresholdMs) || model.thresholdMs <= 0 || !Number.isFinite(model.bias) ||
		!Array.isArray(model.weights) || model.weights.length !== DIMENSIONS || !model.weights.every(Number.isFinite) ||
		!Array.isArray(model.tools) || model.tools.length === 0 || !model.tools.every((tool) => typeof tool === "string") ||
		!["synthetic", "measured", "mixed"].includes(model.training?.basis)) {
		throw new Error("Invalid duration model");
	}
	return model;
}

export function predict(model, call) {
	const vector = features(call);
	const knownTool = model.tools.includes(call.tool);
	const longScore = scoreVector(model, vector);
	return {
		durationClass: !knownTool || (longScore > 0.25 && longScore < 0.75) ? "uncertain" : longScore >= 0.75 ? "long" : "short",
		longScore,
		thresholdMs: model.thresholdMs,
		basis: model.training.basis,
		...(!knownTool ? { reason: "unseen tool" } : {}),
	};
}

export function validateDataset(dataset) {
	if (!dataset || !Number.isFinite(dataset.thresholdMs) || dataset.thresholdMs <= 0 ||
		!Array.isArray(dataset.examples) || dataset.examples.length === 0) {
		throw new Error("Expected positive thresholdMs and nonempty examples");
	}
	const groups = new Map();
	const calls = new Map();
	const examples = dataset.examples.map((example, index) => {
		validateCall(example);
		if (typeof example.group !== "string" || !example.group || !["train", "test"].includes(example.split)) {
			throw new Error(`Example ${index}: group and train/test split required`);
		}
		const measured = Object.hasOwn(example, "durationMs");
		if (measured ? !Number.isFinite(example.durationMs) || example.durationMs < 0 || Object.hasOwn(example, "label") :
			!["short", "long"].includes(example.label)) {
			throw new Error(`Example ${index}: supply either a nonnegative measured durationMs or a synthetic label`);
		}
		if (groups.has(example.group) && groups.get(example.group) !== example.split) {
			throw new Error(`Group ${example.group} leaks across train/test`);
		}
		groups.set(example.group, example.split);
		// Also catch identical feature vectors across splits, including changes beyond the input limit.
		const key = JSON.stringify(features(example).sort(([a], [b]) => a - b));
		if (calls.has(key) && calls.get(key) !== example.split) throw new Error(`Example ${index}: duplicate call across train/test`);
		calls.set(key, example.split);
		return { ...example, label: measured ? (example.durationMs > dataset.thresholdMs ? "long" : "short") : example.label,
			source: measured ? "measured" : "synthetic" };
	});
	for (const split of ["train", "test"]) {
		if (!["short", "long"].every((label) => examples.some((example) => example.split === split && example.label === label))) {
			throw new Error(`Both short and long examples required in ${split}`);
		}
	}
	return { thresholdMs: dataset.thresholdMs, examples };
}

export function train(dataset) {
	const checked = validateDataset(dataset);
	const examples = checked.examples.filter((example) => example.split === "train");
	const vectors = examples.map(features);
	const weights = new Float64Array(DIMENSIONS);
	let bias = 0;
	// Fixed full-batch gradient descent: no randomness, no test-set threshold/hyperparameter fitting.
	const epochs = 500;
	const learningRate = 2;
	const l2 = 0.001;
	for (let epoch = 0; epoch < epochs; epoch++) {
		const gradient = new Float64Array(DIMENSIONS);
		let biasGradient = 0;
		for (let index = 0; index < examples.length; index++) {
			const error = scoreVector({ bias, weights }, vectors[index]) - Number(examples[index].label === "long");
			biasGradient += error;
			for (const [slot, value] of vectors[index]) gradient[slot] += error * value;
		}
		bias -= learningRate * biasGradient / examples.length;
		for (let slot = 0; slot < DIMENSIONS; slot++) {
			weights[slot] -= learningRate * (gradient[slot] / examples.length + l2 * weights[slot]);
		}
	}
	const sources = new Set(examples.map((example) => example.source));
	return {
		algorithm: "hashed-logistic-regression",
		dimensions: DIMENSIONS,
		maxInputCharacters: MAX_INPUT_CHARACTERS,
		thresholdMs: checked.thresholdMs,
		bias: Math.fround(bias),
		weights: Array.from(weights, Math.fround),
		tools: [...new Set(examples.map((example) => example.tool))].sort(),
		training: { examples: examples.length, basis: sources.size > 1 ? "mixed" : [...sources][0], epochs, learningRate, l2 },
	};
}

export function baseline(call) {
	const command = typeof call.arguments.command === "string" ? call.arguments.command.toLowerCase() : "";
	if (/--help|--version|\s-h(?:\s|$)/.test(command)) return "short";
	return /\b(test|build|check|install|ci|clone|fetch|pull|push|compile|lint|sleep|pytest|vitest|jest|tsc)\b/.test(command) ||
		call.tool === "subagent" ? "long" : "short";
}

function metrics(examples, classify) {
	const counts = { longAsLong: 0, longAsShort: 0, shortAsLong: 0, shortAsShort: 0, uncertain: 0 };
	let correct = 0;
	for (const example of examples) {
		const predicted = classify(example);
		if (predicted === "uncertain") { counts.uncertain++; continue; }
		counts[`${example.label}As${predicted === "long" ? "Long" : "Short"}`]++;
		if (predicted === example.label) correct++;
	}
	const decided = examples.length - counts.uncertain;
	return {
		...counts,
		coverage: decided / examples.length,
		accuracyOnDecided: decided ? correct / decided : null,
		longPrecision: counts.longAsLong + counts.shortAsLong ? counts.longAsLong / (counts.longAsLong + counts.shortAsLong) : null,
		longRecall: counts.longAsLong / examples.filter((example) => example.label === "long").length,
	};
}

export function evaluate(model, dataset) {
	validateModel(model);
	const checked = validateDataset(dataset);
	if (checked.thresholdMs !== model.thresholdMs) throw new Error("Dataset and model duration thresholds differ");
	const examples = checked.examples.filter((example) => example.split === "test");
	const sources = new Set(examples.map((example) => example.source));
	return {
		testExamples: examples.length,
		testGroups: new Set(examples.map((example) => example.group)).size,
		testBasis: sources.size > 1 ? "mixed" : [...sources][0],
		thresholdMs: model.thresholdMs,
		modelForcedAtHalf: metrics(examples, (call) => predict(model, call).longScore >= 0.5 ? "long" : "short"),
		modelWithAbstention: metrics(examples, (call) => predict(model, call).durationClass),
		keywordBaseline: metrics(examples, baseline),
		alwaysShortBaseline: metrics(examples, () => "short"),
		errorsAtHalf: examples.filter((call) => (predict(model, call).longScore >= 0.5 ? "long" : "short") !== call.label)
			.map((call) => ({ tool: call.tool, arguments: call.arguments, expected: call.label, ...predict(model, call) })),
	};
}
