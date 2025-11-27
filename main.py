"""
Murim Simulation - M1 Architecture

M1 Features:
- Single monk with hunger/stamina/skill stats
- Three action tiles: kitchen, rest area, training grounds
- Observe → Reflect → Plan → Act brain loop
- Memory module with trend detection and adaptive behavior
"""


class World:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        # Special tiles for M1
        self.kitchen_pos = (2, 2)
        self.rest_pos = (0, 4)
        self.training_pos = (4, 0)

        # Monk position
        self.monk_pos = (0, 0)

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def move_monk(self, x, y):
        """Move the monk to a new position."""
        if not self.is_inside(x, y):
            raise ValueError(f"Invalid move: {(x, y)} outside grid.")
        self.monk_pos = (x, y)

    def show_world(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        kx, ky = self.kitchen_pos
        rx, ry = self.rest_pos
        tx, ty = self.training_pos
        mx, my = self.monk_pos

        grid[ky][kx] = " K"
        grid[ry][rx] = " R"
        grid[ty][tx] = " T"
        grid[my][mx] = " M"

        for row in grid:
            print("".join(row))


class Monk:
    def __init__(self, name="Zhang Wei"):
        self.name = name
        
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        self.busy_ticks = 0
        self.current_action = None
        self.target_tile = None
        self.desire = None

        self.last_thought = ""
        self.position = (0, 0)

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
        self.position = world.monk_pos

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

        # If already at the target tile, start the action
        if (x, y) == (tx, ty):
            if self.current_action is None:
                return self.start_action()

        # Otherwise, move one step toward target
        new_x, new_y = x, y

        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        world.move_monk(new_x, new_y)
        self.stamina = max(0, self.stamina - 1)

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

def game_loop(ticks=50):
    world = World()
    monk = Monk()

    for tick in range(1, ticks + 1):
        print(f"\n--- Tick {tick} ---")

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
            f"skill={monk.skill}"
        )
        
        # Show grid every 10 ticks
        if tick % 10 == 0:
            print()
            world.show_world()


if __name__ == "__main__":
    game_loop(ticks=50)
