import assert from "node:assert/strict";
import { test } from "node:test";
import register from "./volt-extension.js";

function harness() {
	const handlers = new Map();
	const observations = [];
	let time = 0;
	let flushed = false;
	const learner = {
		begin: (call) => ({ call }),
		complete: async (ticket, result) => { observations.push({ ticket, ...result }); },
		refresh: async () => {},
		flush: async () => { flushed = true; },
	};
	const controller = new AbortController();
	const context = { cwd: "/repo", signal: controller.signal, hasUI: true, ui: { notify: () => {} } };
	register({ on: (name, handler) => handlers.set(name, handler) }, {
		open: async (cwd) => { assert.equal(cwd, "/repo"); return learner; }, now: () => time,
	});
	return { observations, controller, learner, context, tick: (ms) => { time += ms; },
		emit: (name, event = {}) => handlers.get(name)(event, context), flushed: () => flushed };
}

test("Volt observer measures foreground bash spans and flushes on shutdown without returning overrides", async () => {
	const host = harness();
	await host.emit("session_start");
	assert.equal(await host.emit("tool_execution_start", { toolCallId: "one", toolName: "bash", args: { command: "npm test" } }), undefined);
	host.tick(2400);
	assert.equal(await host.emit("tool_execution_end", { toolCallId: "one", result: {}, isError: false }), undefined);
	assert.equal(host.observations[0].durationMs, 2400);
	assert.equal(host.observations[0].status, "completed");
	await host.emit("session_shutdown");
	assert.equal(host.flushed(), true);
});

test("background launch acknowledgements and non-bash tools never become training samples", async () => {
	const host = harness();
	await host.emit("session_start");
	await host.emit("tool_execution_start", { toolCallId: "background", toolName: "bash", args: { command: "npm test", background: true } });
	await host.emit("tool_execution_end", { toolCallId: "background", result: {} });
	await host.emit("tool_execution_start", { toolCallId: "read", toolName: "read", args: { path: "file" } });
	await host.emit("tool_execution_end", { toolCallId: "read", result: {} });
	await host.emit("tool_execution_start", { toolCallId: "wrapped", toolName: "bash", args: { command: "npm test" } });
	await host.emit("tool_execution_end", { toolCallId: "wrapped", result: { details: { backgroundJob: { id: "job" } } } });
	assert.equal(host.observations.length, 0);
});

test("failures and cancellation are marked separately and abandoned executions are forgotten", async () => {
	const host = harness();
	await host.emit("session_start");
	await host.emit("tool_execution_start", { toolCallId: "failed", toolName: "bash", args: { command: "bad" } });
	await host.emit("tool_execution_end", { toolCallId: "failed", result: {}, isError: true });
	assert.equal(host.observations[0].status, "failed");
	await host.emit("tool_execution_start", { toolCallId: "cancelled", toolName: "bash", args: { command: "sleep 30" } });
	host.controller.abort();
	await host.emit("tool_execution_end", { toolCallId: "cancelled", result: {}, isError: true });
	assert.equal(host.observations[1].status, "cancelled");
	await host.emit("tool_execution_start", { toolCallId: "abandoned", toolName: "bash", args: { command: "npm test" } });
	await host.emit("agent_end");
	await host.emit("tool_execution_end", { toolCallId: "abandoned", result: {} });
	assert.equal(host.observations.length, 2);
});

test("observer storage failures warn once and do not fail tool execution", async () => {
	const host = harness();
	const warnings = [];
	host.context.ui.notify = (message) => warnings.push(message);
	host.learner.complete = async () => { throw new Error("disk unavailable"); };
	await host.emit("session_start");
	for (const id of ["one", "two"]) {
		await host.emit("tool_execution_start", { toolCallId: id, toolName: "bash", args: { command: "pwd" } });
		await host.emit("tool_execution_end", { toolCallId: id, result: {} });
	}
	assert.equal(warnings.length, 1);
	assert.match(warnings[0], /disk unavailable/);
});
