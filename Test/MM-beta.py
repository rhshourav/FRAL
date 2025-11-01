from pynput import mouse
import time
import threading

# Time (in seconds) to detect inactivity
INACTIVITY_LIMIT = 10

last_move_time = time.time()

def on_move(x, y):
    global last_move_time
    last_move_time = time.time()
    # Optional: print current position
    # print(f"Mouse moved to ({x}, {y})")

def check_inactivity():
    while True:
        if time.time() - last_move_time > INACTIVITY_LIMIT:
            print("Mouse has not moved for 10 seconds!")
            # Wait until next movement before printing again
            while time.time() - last_move_time > INACTIVITY_LIMIT:
                time.sleep(1)
        time.sleep(1)

# Start the inactivity checker in a separate thread
threading.Thread(target=check_inactivity, daemon=True).start()

# Start mouse listener
with mouse.Listener(on_move=on_move) as listener:
    listener.join()
