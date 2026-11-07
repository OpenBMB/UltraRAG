/** Share pending work for one key, retaining neither successes nor failures. */
export class SingleFlight<T> {
  private pending = new Map<string, Promise<T>>();

  run(key: string, action: () => Promise<T>): Promise<T> {
    const existing = this.pending.get(key);
    if (existing) return existing;
    const task = Promise.resolve().then(action).finally(() => {
      if (this.pending.get(key) === task) this.pending.delete(key);
    });
    this.pending.set(key, task);
    return task;
  }
}
