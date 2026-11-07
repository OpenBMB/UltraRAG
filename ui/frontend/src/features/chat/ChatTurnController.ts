export type ChatTurn = { id: number; controller: AbortController };

/** Own one foreground turn until both its stream and server stop have settled. */
export class ChatTurnController {
  private sequence = 0;
  private active: ChatTurn | null = null;
  private stopping: Promise<void> | null = null;

  get busy(): boolean {
    return this.active !== null;
  }

  begin(): ChatTurn {
    if (this.busy) throw new Error("A chat turn is already active");
    const turn = { id: ++this.sequence, controller: new AbortController() };
    this.active = turn;
    return turn;
  }

  isCurrent(turn: ChatTurn): boolean {
    return this.active === turn && !turn.controller.signal.aborted;
  }

  finish(turn: ChatTurn): void {
    if (this.isCurrent(turn)) this.active = null;
  }

  async stop(stopServer: () => Promise<unknown>): Promise<void> {
    if (this.stopping) return this.stopping;
    const turn = this.active;
    if (!turn) return;
    turn.controller.abort();
    this.stopping = Promise.resolve()
      .then(stopServer)
      .then(() => undefined)
      .finally(() => {
        if (this.active === turn) this.active = null;
        this.stopping = null;
      });
    return this.stopping;
  }
}
