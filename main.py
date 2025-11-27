import random

class World:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        # Shared resource — key M1.5 tile
        self.training_pos = (2, 2)

        # Other tiles
        self.kitchen_pos = (0, 4)
        self.rest_pos = (4, 0)

        # Monk positions stored in dict {name: (x,y)}
        self.monk_positions = {}

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def register_monk(self, monk_name, pos):
        self.monk_positions[monk_name] = pos

    def move_monk(self, monk_name, new_pos):
        x, y = new_pos
        if not self.is_inside(x, y):
            raise ValueError(f"{monk_name} tries to move outside grid: {new_pos}")
        self.monk_positions[monk_name] = (x, y)

    def is_tile_free(self, pos, ignore=None):
        """Check if tile is free. Optionally ignore one monk (the one asking)."""
        for name, p in self.monk_positions.items():
            if ignore is not None and name == ignore:
                continue
            if p == pos:
                return False
        return True

    def who_is_at(self, pos):
        for name, p in self.monk_positions.items():
            if p == pos:
                return name
        return None

    def print_world(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        # Mark special tiles
        tx, ty = self.training_pos
        kx, ky = self.kitchen_pos
        rx, ry = self.rest_pos

        grid[ty][tx] = " T"
        grid[ky][kx] = " K"
        grid[ry][rx] = " R"

        # Mark monks
        for name, (mx, my) in self.monk_positions.items():
            grid[my][mx] = " M"

        for row in grid:
            print("".join(row))


class Monk:
    def __init__(self, name):
        self.name = name

        # Basic stats
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        # Action engine
        self.busy_ticks = 0
        self.current_action = None
        self.desire = None
        self.target_tile = None

        # M1.5 conflict tracking
        self.conflict_count = 0

        # Memory
        self.last_thought = ""
        self.memory = {
            "hunger_trend": [],
            "stamina_trend": [],
            "success_streak": 0,
            "failure_streak": 0,
        }

        self.position = (0, 0)

    # ---------------- TICK PIPELINE ---------------- #

    def observe(self, world):
        self.position = world.monk_positions[self.name]

    def update_memory(self):
        # Track hunger/stamina trends
        trend = self.memory["hunger_trend"]
        trend.append(self.hunger)
        if len(trend) > 5:
            trend.pop(0)

        trend = self.memory["stamina_trend"]
        trend.append(self.stamina)
        if len(trend) > 5:
            trend.pop(0)

    def reflect(self):
        # Yield behavior if blocked too long
        if self.conflict_count > 2:
            self.last_thought = (
                f"The way remains blocked… I should choose patience and rest."
            )
            self.desire = "rest"
            return "rest"

        # Default need logic
        if self.hunger > 70:
            self.last_thought = f"I feel hunger rising."
            self.desire = "eat"
            return "eat"

        if self.stamina < 30:
            self.last_thought = f"My strength fades."
            self.desire = "rest"
            return "rest"

        self.last_thought = f"I should train."
        self.desire = "train"
        return "train"

    def plan(self, world):
        if self.desire == "eat":
            self.target_tile = world.kitchen_pos
        elif self.desire == "rest":
            self.target_tile = world.rest_pos
        else:
            self.target_tile = world.training_pos

    def act(self, world):
        # ---------------- BUSY: Continue or Abort ---------------- #
        if self.busy_ticks > 0:
            # Emergency abort
            if self.hunger > 90 or self.stamina < 10:
                if self.current_action not in ["eating", "resting"]:
                    self._record_failure()
                    self.busy_ticks = 0
                    self.current_action = None
                    self.conflict_count = 0
                    return f"{self.name} falters and stops the action."

            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                return self.finish_action()

            return f"{self.name} continues {self.current_action}."

        # ---------------- NOT BUSY: Movement or Start Action ---------------- #

        x, y = self.position
        tx, ty = self.target_tile

        # If on target tile
        if (x, y) == (tx, ty):
            # But if another monk occupies training tile
            if (
                self.desire == "train"
                and not world.is_tile_free(world.training_pos, ignore=self.name)
            ):
                self.conflict_count += 1
                return f"{self.name} bows. Another monk uses the training ground. I wait."
            return self.start_action()

        # Move toward tile
        new_x, new_y = x, y

        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        # If tile is occupied → conflict
        if not world.is_tile_free((new_x, new_y), ignore=self.name):
            self.conflict_count += 1
            blocker = world.who_is_at((new_x, new_y))
            return f"{self.name} pauses. {blocker} stands in my path. I wait."

        # Movement succeeds
        self.conflict_count = 0
        world.move_monk(self.name, (new_x, new_y))
        self.stamina = max(0, self.stamina - 1)

        direction = self._dir((x, y), (new_x, new_y))
        return f"{self.name}: {self.last_thought} The path lies {direction}. I walk."

    # ---------------- ACTIONS ---------------- #

    def start_action(self):
        self.conflict_count = 0

        if self.desire == "eat":
            self.current_action = "eating"
            self.busy_ticks = 3
            return f"{self.name} begins eating."

        if self.desire == "rest":
            self.current_action = "resting"
            self.busy_ticks = 5
            return f"{self.name} lies down to rest."

        self.current_action = "training"
        self.busy_ticks = 4
        return f"{self.name} begins training."

    def finish_action(self):
        if self.current_action == "eating":
            self.hunger = max(0, self.hunger - 30)
            self.stamina = min(100, self.stamina + 10)
            msg = f"{self.name} finishes eating."

        elif self.current_action == "resting":
            self.stamina = min(100, self.stamina + 40)
            self.hunger = min(100, self.hunger + 5)
            msg = f"{self.name} rises refreshed."

        else:
            self.skill += 5
            self.stamina = max(0, self.stamina - 20)
            self.hunger = min(100, self.hunger + 10)
            msg = f"{self.name}'s training sharpens his skill."

        self._record_success()
        self.current_action = None
        self.target_tile = None
        return msg

    # ---------------- MEMORY HELPERS ---------------- #

    def _record_success(self):
        self.memory["success_streak"] += 1
        self.memory["failure_streak"] = 0

    def _record_failure(self):
        self.memory["failure_streak"] += 1
        self.memory["success_streak"] = 0

    def _dir(self, old, new):
        ox, oy = old
        nx, ny = new
        if nx > ox: return "east"
        if nx < ox: return "west"
        if ny > oy: return "south"
        return "north"


# ---------------- SIMULATION ---------------- #

def game_loop(ticks=50):
    world = World()

    # Two monks for M1.5
    monks = [
        Monk("Chung Myung"),
        Monk("Chung Li"),
    ]

    # Opposite corners — guarantees natural collision
    world.register_monk(monks[0].name, (0, 0))
    world.register_monk(monks[1].name, (4, 4))

    for tick in range(1, ticks + 1):
        print(f"\n===== TICK {tick} =====")

        for monk in monks:
            monk.observe(world)
            monk.update_memory()
            monk.reflect()
            monk.plan(world)
            log = monk.act(world)
            monk.hunger = min(100, monk.hunger + 1)
            print(log)

        # Uncomment if you want visual grid
        # world.print_world()


if __name__ == "__main__":
    game_loop()
