import time
import rrr

w = rrr.MemoryWatcher(spawn_dosbox=True)
time.sleep(5)
# Wait for game state
for _ in range(100):
    s = w.read_state()
    if s is not None:
        break
    time.sleep(0.02)
else:
    raise RuntimeError("No game state")

print("Connected. Testing actions 0..18")

for a in range(19):
    print("action", a)
    for b in range(100):
        w.apply_action(a)
        time.sleep(0.02)
        w.release_action()
print("Done")
