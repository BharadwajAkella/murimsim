"""
Murim Simulation - M1.5 Architecture

M1 Features:
- Single monk with hunger/stamina/skill stats
- Three action tiles: kitchen, rest area, training grounds
- Observe → Reflect → Plan → Act brain loop
- Memory module with trend detection and adaptive behavior

M1.5 Features (Multi-Monk):
- Multiple monks in shared world
- Tile occupancy tracking
- Conflict detection (blocking and waiting)
- Per-monk conflict counters
- Clean separation: World manages positions, Monks manage behavior
"""


class World:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        # Special tiles for M1
        self.kitchen_pos = (2, 2)
        self.rest_pos = (0, 4)
        self.training_pos = (4, 0)

        # Multi-monk support (M1.5)
        self.monks = {}  # {monk_id: position}
        self.tile_occupants = {}  # {position: monk_id}

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def register_monk(self, monk_id, start_pos=(0, 0)):
        """Register a monk in the world."""
        if not self.is_inside(*start_pos):
            raise ValueError(f"Invalid position: {start_pos}")
        self.monks[monk_id] = start_pos
        self.tile_occupants[start_pos] = monk_id

    def move_monk(self, monk_id, x, y):
        """Move a monk to a new position."""
        if not self.is_inside(x, y):
            raise ValueError(f"Invalid move: {(x, y)} outside grid.")
        
        # Clear old position
        old_pos = self.monks.get(monk_id)
        if old_pos and old_pos in self.tile_occupants:
            del self.tile_occupants[old_pos]
        
        # Set new position
        self.monks[monk_id] = (x, y)
        self.tile_occupants[(x, y)] = monk_id

    def get_monk_position(self, monk_id):
        """Get a monk's current position."""
        return self.monks.get(monk_id)

    def is_tile_occupied(self, pos):
        """Check if a tile is occupied by any monk."""
        return pos in self.tile_occupants

    def get_occupant(self, pos):
        """Get the monk_id occupying a tile, or None."""
        return self.tile_occupants.get(pos)

    def show_world(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        kx, ky = self.kitchen_pos
        rx, ry = self.rest_pos
        tx, ty = self.training_pos

        grid[ky][kx] = " K"
        grid[ry][rx] = " R"
        grid[ty][tx] = " T"

        # Display all monks
        for monk_id, (mx, my) in self.monks.items():
            # Use numbers for monk IDs (1, 2, etc.)
            grid[my][mx] = f" {monk_id}"

        for row in grid:
            print("".join(row))


class Monk:
    def __init__(self, monk_id, name=None):
        self.monk_id = monk_id
        self.name = name or f"Monk {monk_id}"
        
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        self.busy_ticks = 0
        self.current_action = None
        self.target_tile = None
        self.desire = None

        self.last_thought = ""
        self.position = (0, 0)
        
        # M1.5 conflict tracking
        self.blocked_by = None  # monk_id of blocker
        self.conflict_count = 0

        # Memory module
        self.memory = {
            "last_action": None,
            "last_position": None,
            "hunger_trend": [],
            "stamina_trend": [],
            "success_streak": 0,
            "failure_streak": 0,
        }

    # ------------------ MONK BRAIN ------------------

    def observe(self, world):
        self.position = world.get_monk_position(self.monk_id)

    def update_memory(self):
        """Update memory at the start of each tick, based on last tick."""
        self.memory["last_action"] = self.current_action
        self.memory["last_position"] = self.position

        # Track trends (max 5 entries)
        self.memory["hunger_trend"].append(self.hunger)
        if len(self.memory["hunger_trend"]) > 5:
            self.memory["hunger_trend"].pop(0)

        self.memory["stamina_trend"].append(self.stamina)
        if len(self.memory["stamina_trend"]) > 5:
            self.memory["stamina_trend"].pop(0)

    def reflect(self):
        # Analyze trends
        hunger_spiking = self._is_hunger_spiking()
        stamina_declining = self._is_stamina_declining()

        # Memory-based early triggers
        if hunger_spiking:
            self.last_thought = (
                "My hunger grows quicker than before... I should eat sooner."
            )
            self.desire = "eat"
            return "eat"

        if stamina_declining:
            self.last_thought = (
                "My strength fades faster than expected... I must rest."
            )
            self.desire = "rest"
            return "rest"

        # Streak-based behavior
        if self.memory["success_streak"] >= 3 and self.stamina > 40:
            self.last_thought = "Training has been fruitful lately. I feel momentum."
            self.desire = "train"
            return "train"

        if self.memory["failure_streak"] >= 2:
            self.last_thought = (
                "I have struggled recently. Perhaps rest will restore my focus."
            )
            self.desire = "rest"
            return "rest"

        # Default logic with adjusted thresholds
        if self.hunger > 70:
            self.last_thought = "I seek food."
            self.desire = "eat"
            return "eat"
        elif self.stamina < 30:
            self.last_thought = "I need rest."
            self.desire = "rest"
            return "rest"
        else:
            self.last_thought = "I should train."
            self.desire = "train"
            return "train"

    def plan(self, desire, world):
        if desire == "eat":
            self.target_tile = world.kitchen_pos
        elif desire == "rest":
            self.target_tile = world.rest_pos
        else:
            self.target_tile = world.training_pos

    def act(self, world):
        x, y = self.position
        tx, ty = self.target_tile

        # If busy, continue action or abort if emergency
        if self.busy_ticks > 0:
            if self.hunger > 90 or self.stamina < 10:
                # Abort any non-recovering action
                if self.current_action not in ["eating", "resting"]:
                    self._record_failure()
                    self.current_action = None
                    self.busy_ticks = 0
                    self.target_tile = None
                    return "Weakness floods my body. I must stop."

            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                return self.finish_action()

            return f"I continue {self.current_action}."

        # If already at the target tile, check for conflict
        if (x, y) == (tx, ty):
            # M1.5: Check if tile is occupied by another monk
            occupant = world.get_occupant((tx, ty))
            if occupant and occupant != self.monk_id:
                self.blocked_by = occupant
                self.conflict_count += 1
                return f"Another monk occupies this place. I must wait."
            
            if self.current_action is None:
                self.blocked_by = None  # Clear any previous blocking
                return self.start_action()

        # Otherwise, move one step toward target
        new_x, new_y = x, y

        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        # M1.5: Check if destination is blocked
        if world.is_tile_occupied((new_x, new_y)):
            occupant = world.get_occupant((new_x, new_y))
            self.blocked_by = occupant
            self.conflict_count += 1
            return f"Another monk blocks my path. I wait."

        world.move_monk(self.monk_id, new_x, new_y)
        self.stamina = max(0, self.stamina - 1)
        self.blocked_by = None  # Clear blocking

        direction = self.direction_to_target((new_x, new_y), self.target_tile)
        return (
            f"{self.last_thought} {self.target_label()} lies {direction}. "
            "I will walk."
        )

    # ------------------ ACTIONS ------------------

    def start_action(self):
        if self.desire == "eat":
            self.current_action = "eating"
            self.busy_ticks = 3
            return "I stand at the kitchen and begin eating."

        elif self.desire == "rest":
            self.current_action = "resting"
            self.busy_ticks = 5
            return "I lie down at the resting mat and begin to rest."

        else:
            self.current_action = "training"
            self.busy_ticks = 4
            return "I step onto the training grounds and begin training."

    def finish_action(self):
        if self.current_action == "eating":
            self.hunger = max(0, self.hunger - 30)
            self.stamina = min(100, self.stamina + 10)
            msg = "I finish eating and feel nourished."

        elif self.current_action == "resting":
            self.stamina = min(100, self.stamina + 40)
            self.hunger = min(100, self.hunger + 5)
            msg = "My strength returns as I finish resting."

        else:  # training
            self.skill += 5
            self.stamina = max(0, self.stamina - 20)
            self.hunger = min(100, self.hunger + 10)
            msg = "I finish training and feel my skill sharpen."

        self._record_success()
        self.current_action = None
        self.target_tile = None
        return msg

    # ------------------ MEMORY HELPERS ------------------

    def _record_success(self):
        self.memory["success_streak"] += 1
        self.memory["failure_streak"] = 0

    def _record_failure(self):
        self.memory["failure_streak"] += 1
        self.memory["success_streak"] = 0

    def _is_hunger_spiking(self):
        """Detect if hunger is rising fast."""
        trend = self.memory["hunger_trend"]
        if len(trend) < 3:
            return False

        deltas = [trend[i] - trend[i - 1] for i in range(1, len(trend))]
        avg_delta = sum(deltas) / len(deltas)

        # Spiking if average increase > 2 per tick and current hunger > 40
        return avg_delta > 2 and self.hunger > 40

    def _is_stamina_declining(self):
        """Detect if stamina is dropping fast."""
        trend = self.memory["stamina_trend"]
        if len(trend) < 3:
            return False

        deltas = [trend[i] - trend[i - 1] for i in range(1, len(trend))]
        avg_delta = sum(deltas) / len(deltas)

        # Declining if average decrease < -4 per tick and stamina < 50
        return avg_delta < -4 and self.stamina < 50

    # ------------------ LOG HELPERS ------------------

    def target_label(self):
        if self.desire == "eat":
            return "The kitchen"
        if self.desire == "rest":
            return "The resting mat"
        return "The training grounds"

    def direction_to_target(self, current_pos, target_pos):
        cx, cy = current_pos
        tx, ty = target_pos
        if tx > cx:
            return "to the east"
        if tx < cx:
            return "to the west"
        if ty > cy:
            return "to the south"
        return "to the north"


# ------------------ MAIN LOOP ------------------

def game_loop(ticks=50, num_monks=1):
    world = World()
    monks = []
    
    # Create monks with different starting positions
    start_positions = [(0, 0), (4, 4), (0, 4), (4, 0)]
    for i in range(num_monks):
        monk_id = i + 1
        monk = Monk(monk_id, name=f"Zhang {['Wei', 'Li', 'Chen', 'Wu'][i % 4]}")
        start_pos = start_positions[i % len(start_positions)]
        world.register_monk(monk_id, start_pos)
        monks.append(monk)

    for tick in range(1, ticks + 1):
        print(f"\n--- Tick {tick} ---")

        # Each monk acts
        for monk in monks:
            monk.observe(world)
            monk.update_memory()
            desire = monk.reflect()
            monk.plan(desire, world)
            log = monk.act(world)

            # Passive hunger increase
            monk.hunger = min(100, monk.hunger + 1)

            print(f"[{monk.name}] {log}")
            print(
                f"  Stats: hunger={monk.hunger}, stamina={monk.stamina}, "
                f"skill={monk.skill}, conflicts={monk.conflict_count}"
            )
        
        # Show grid every 5 ticks or when conflicts occur
        if tick % 5 == 0 or any(m.conflict_count > 0 and m.blocked_by for m in monks):
            print()
            world.show_world()


if __name__ == "__main__":
    # M1: Single monk
    # game_loop(ticks=50, num_monks=1)
    
    # M1.5: Two monks with conflict
    game_loop(ticks=50, num_monks=2)
