import os
import subprocess
import tempfile
import numpy as np
import json
import atexit
import threading
import time

class StormContainerManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.container = None
        self.temp_dir = None
        self._initialized = True
        
        # Register cleanup on program exit
        atexit.register(self.cleanup)
    
    def start_container(self):
        if self.container is not None:
            return  # Already running
            
        # Create persistent temp directory
        self.temp_dir = tempfile.mkdtemp()
        #print(f"DEBUG: Created persistent temp dir: {self.temp_dir}")
        
        # Start container with interactive shell
        cmd = [
            "docker", "run", "-i", "--rm", 
            "-v", f"{self.temp_dir}:/data",
            "-w", "/data",
            "movesrwth/storm:stable", 
            "bash"
        ]
        
        #print(f"DEBUG: Starting persistent container: {' '.join(cmd)}")
        self.container = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Test if container is ready
        self._send_command("echo 'Container ready'")
        #print("DEBUG: Container started and ready")
    
    def _send_command(self, command):
        if self.container is None:
            raise RuntimeError("Container not started")
            
        #print(f"DEBUG: Sending command: {command}")
        self.container.stdin.write(command + "\n")
        self.container.stdin.flush()
        
        # Read until we get a result or error
        output_lines = []
        while True:
            line = self.container.stdout.readline()
            if not line:
                break
            output_lines.append(line.strip())
            #print(f"DEBUG: Container output: {line.strip()}")
            
            # Look for completion indicators
            if "Result" in line or "ERROR" in line or "Container ready" in line:
                break
                
        return "\n".join(output_lines)
    
    def run_storm(self, jani_filename, goal_state):
        self.start_container()  # Ensure container is running
        
        storm_cmd = f"storm --jani {jani_filename} --prop 'Pmax=? [F pos={goal_state}]'"
        output = self._send_command(storm_cmd)
        
        # Parse result
        for line in output.splitlines():
            if "Result" in line:
                try:
                    prob = float(line.split()[-1])
                    #print(f"DEBUG: Extracted probability: {prob}")
                    return prob
                except (ValueError, IndexError):
                    continue
                    
        raise RuntimeError(f"Failed to extract probability from Storm output: {output}")
    
    def cleanup(self):
        #print("DEBUG: Cleaning up Storm container")
        if self.container:
            try:
                self.container.stdin.close()
                self.container.terminate()
                self.container.wait(timeout=5)
            except:
                self.container.kill()
            self.container = None
            
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
            self.temp_dir = None

# Global instance
storm_manager = StormContainerManager()

def compute_win_probability(desc: np.ndarray, slip: list[float]) -> float:
    #print("DEBUG: Starting compute_win_probability")
    size = desc.shape[0]
    #print(f"DEBUG: Grid size: {size}x{size}")

    def to_state(i, j):
        return i * size + j

    initial_state = to_state(0, 0)
    goal_state = to_state(size - 1, size - 1)
    #print(f"DEBUG: Initial state: {initial_state}, Goal state: {goal_state}")
    #print(f"DEBUG: Slip input: {slip}")

    #print("DEBUG: Building JANI model")
    jani_model = build_jani_model(desc, slip, size, goal_state, initial_state)

    # Ensure container is started first to get temp_dir
    storm_manager.start_container()
    
    # Write JANI file to persistent temp directory
    jani_filename = f"model_{int(time.time()*1000000)}.jani"
    jani_path = os.path.join(storm_manager.temp_dir, jani_filename)
    
    with open(jani_path, 'w') as f:
        json.dump(jani_model, f)
    #print(f"DEBUG: Wrote JANI model to: {jani_path}")

    # Use persistent container
    return storm_manager.run_storm(jani_filename, goal_state)


def build_jani_model(desc, slip, size, goal_state, initial_state):
    def to_state(i, j):
        return i * size + j

    move = {
        0: (0, -1),  # LEFT
        1: (1, 0),   # DOWN
        2: (0, 1),   # RIGHT
        3: (-1, 0),  # UP
    }

    slip_deltas = {
        0: [move[0], move[3], move[1]],
        1: [move[1], move[0], move[2]],
        2: [move[2], move[1], move[3]],
        3: [move[3], move[2], move[0]],
    }

    action = 1
    deltas = slip_deltas[action]
    #print(f"DEBUG: Using action={action} with deltas={deltas} and slip={slip}")

    transitions = []
    for i in range(size):
        for j in range(size):
            s = to_state(i, j)
            tile = desc[i][j]

            if tile == b'H' or tile == b'G':
                target = {"location": "loc0", "assignments": [{"ref": "pos", "value": s}]}
                transitions.append({
                    "location": "loc0",
                    "guard": {"exp": {"op": "=", "left": "pos", "right": s}},
                    "destinations": [{"location": "loc0", "probability": {"exp": 1.0}, "assignments": [{"ref": "pos", "value": s}]}]
                })
            else:
                destinations = []
                for p, (dx, dy) in zip(slip, deltas):
                    ni, nj = i + dx, j + dy
                    ns = to_state(ni, nj) if (0 <= ni < size and 0 <= nj < size) else s
                    destinations.append({
                        "location": "loc0",
                        "probability": {"exp": p},
                        "assignments": [{"ref": "pos", "value": ns}]
                    })
                transitions.append({
                    "location": "loc0",
                    "guard": {"exp": {"op": "=", "left": "pos", "right": s}},
                    "destinations": destinations
                })

    return {
        "jani-version": 1,
        "name": "FrozenLakeModel",
        "type": "mdp",
        "features": ["derived-operators"],
        "variables": [
            {
                "name": "pos",
                "type": {
                    "kind": "bounded",
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": size * size - 1
                },
                "initial-value": initial_state
            }
        ],
        "automata": [
            {
                "name": "lake",
                "locations": [{"name": "loc0"}],
                "initial-locations": ["loc0"],
                "edges": transitions
            }
        ],
        "system": {
            "elements": [{"automaton": "lake"}]
        },
        "properties": [
            {
                "name": "reach_goal",
                "expression": {
                    "op": "filter",
                    "fun": "max",
                    "values": {
                        "op": "Pmax",
                        "exp": {
                            "op": "F",
                            "exp": {"op": "=", "left": "pos", "right": goal_state}
                        }
                    },
                    "states": {"op": "initial"}
                }
            }
        ]
    }


def approximate_win_probability(desc: np.ndarray, slip: list[float]) -> float:
    size = desc.shape[0]
    hole_ratio = np.sum(desc == b'H') / (size * size)
    slip_penalty = slip[1] + slip[2]  # penalty for lateral slips
    return max(0.0, min(1.0, 1.0 - (hole_ratio + 0.5 * slip_penalty) ** 1.5))


# Manual cleanup function if needed
def cleanup_storm_container():
    """Call this manually if you want to force cleanup before program exit"""
    storm_manager.cleanup()