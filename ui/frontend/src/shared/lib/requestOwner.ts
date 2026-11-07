/** Keep stale browser requests from settling a newer view's state. */
export class RequestOwner {
  private active: AbortController | null = null;

  get busy(): boolean { return this.active !== null; }

  begin(): AbortController {
    this.cancel();
    const controller = new AbortController();
    this.active = controller;
    return controller;
  }

  owns(controller: AbortController): boolean { return this.active === controller; }

  finish(controller: AbortController): boolean {
    if (!this.owns(controller)) return false;
    this.active = null;
    return true;
  }

  abort(): void { this.active?.abort(); }

  cancel(): void {
    const controller = this.active;
    this.active = null;
    controller?.abort();
  }
}
