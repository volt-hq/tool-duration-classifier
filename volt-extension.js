import { performance } from "node:perf_hooks";
import { openRepository } from "./online.mjs";

// Observation only. Never changes arguments, tool results, or scheduling decisions.
export default function register(volt, { open = openRepository, now = () => performance.now() } = {}) {
	let learner;
	let warned = false;
	const running = new Map();
	const warn = (error, ctx) => {
		if (warned) return;
		warned = true;
		const message = `Tool duration observer: ${error.message}`;
		if (ctx.hasUI) ctx.ui.notify(message, "warning");
		else console.error(message);
	};
	volt.on("session_start", async (_event, ctx) => {
		try { learner = await open(ctx.cwd); }
		catch (error) { warn(error, ctx); }
	});
	volt.on("agent_start", async (_event, ctx) => {
		try { await learner?.refresh(); }
		catch (error) { warn(error, ctx); }
	});
	volt.on("tool_execution_start", (event, ctx) => {
		if (!learner || event.toolName !== "bash" || event.args.background === true) return;
		try {
			const ticket = learner.begin({ tool: event.toolName, arguments: event.args });
			running.set(event.toolCallId, { ticket, startedAt: now(), signal: ctx.signal });
		} catch (error) { warn(error, ctx); }
	});
	volt.on("tool_execution_end", (event, ctx) => {
		const entry = running.get(event.toolCallId);
		if (!entry) return;
		running.delete(event.toolCallId);
		// The launch acknowledgement is not the background task's completed duration.
		if (event.result?.details?.backgroundJob) return;
		const status = entry.signal?.aborted || ctx.signal?.aborted ? "cancelled" :
			event.isError || event.result?.isError ? "failed" : "completed";
		void learner.complete(entry.ticket, { durationMs: now() - entry.startedAt, status })
			.catch((error) => warn(error, ctx));
	});
	volt.on("agent_end", () => { running.clear(); });
	volt.on("session_shutdown", async () => {
		running.clear();
		await learner?.flush();
		learner = undefined;
	});
}
