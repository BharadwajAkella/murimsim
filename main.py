import random


class World:
    def __init__(self, width: int = 5, height: int = 5) -> None:
        self.width = width
        self.height = height

        # Shared training ground (single resource)
        self.training_pos = (2, 2)

        # Per-monk private tiles (for eating/resting, no contention here)
        self.tiles = {
            "eat_A": (0, 4),
            "rest_A": (0, 3),

            "eat_B": (4, 0),
            "rest_B": (4, 1),
        }

        # Current positions of all monks
        self.monk_positions: dict[str, tuple[int, int]] = {}

    def is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def register_monk(self, name: str, pos: tuple[int, int]) -> None:
        self.monk_positions[name] = pos

    def move_monk(self, name: str, new_pos: tuple[int, int]) -> None:
        x, y = new_pos
        if not self.is_inside(x, y):
            raise ValueError(f"{name} attempted to move off-grid: {new_pos}")
        self.monk_positions[name] = (x, y)

    def is_tile_free(self, pos: tuple[int, int],
                     ignore: str | None = None) -> bool:
        for name, p in self.monk_positions.items():
            if ignore is not None and name == ignore:
                continue
            if p == pos:
                return False
        return True

    def who_is_at(self, pos: tuple[int, int]) -> str | None:
        for name, p in self.monk_positions.items():
            if p == pos:
                return name
        return None

    def print_world(self) -> None:
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        tx, ty = self.training_pos
        grid[ty][tx] = " T"

        for name, (mx, my) in self.monk_positions.items():
            grid[my][mx] = " M"

        for row in grid:
            print("".join(row))


class Monk:
    def __init__(self, name: str, personality: dict[str, float],
                 tile_prefix: str) -> None:
        self.name = name

        # Basic stats
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        # Personality weights
        self.personality = personality
        self.tile_prefix = tile_prefix

        # Action state
        self.busy_ticks = 0
        self.current_action: str | None = None
        self.desire: str | None = None
        self.target_tile: tuple[int, int] | None = None

        # Scheduling state (M3)
        self.state = "idle"  # idle / moving / queued / acting
        self.waiting_tile: tuple[int, int] | None = None

        # Memory (simple trends)
        self.memory = {
            "hunger_trend": [],
            "stamina_trend": [],
        }

        self.last_thought = ""
        self.position = (0, 0)

    # ----------------- TICK PIPELINE ----------------- #

    def observe(self, world: World) -> None:
        self.position = world.monk_positions[self.name]

    def update_memory(self) -> None:
        self.memory["hunger_trend"].append(self.hunger)
        if len(self.memory["hunger_trend"]) > 5:
            self.memory["hunger_trend"].pop(0)

        self.memory["stamina_trend"].append(self.stamina)
        if len(self.memory["stamina_trend"]) > 5:
            self.memory["stamina_trend"].pop(0)

    def reflect(self) -> None:
        fp = self.personality["food_priority"]
        rp = self.personality["rest_priority"]
        td = self.personality["training_drive"]

        # If currently queued and personality says "nah", abandon queue
        if self.state == "queued":
            hunger_pressure = self.hunger * fp
            fatigue_pressure = (100 - self.stamina) * rp
            # Simple rule: if either pressure is very high, give up queue
            if hunger_pressure > 80 or fatigue_pressure > 70:
                self.last_thought = (
                    "Waiting tests my body. I should tend to my needs first."
                )
                # choose whether to eat or rest based on which is worse
                if hunger_pressure >= fatigue_pressure:
                    self.desire = "eat"
                else:
                    self.desire = "rest"
                self.state = "idle"
                self.waiting_tile = None
                return

        # Default personality-driven decision
        hunger_pressure = self.hunger * fp
        fatigue_pressure = (100 - self.stamina) * rp

        if hunger_pressure > 70:
            self.last_thought = "I should eat before hunger distracts me."
            self.desire = "eat"
            return

        if fatigue_pressure > 40:
            self.last_thought = "My body seeks rest."
            self.desire = "rest"
            return

        # Training by default, influenced by training_drive (td)
        # We don't need td in formula; it's implicit in how “sticky”
        # you make training in future milestones.
        self.last_thought = "Training will sharpen my mind and body."
        self.desire = "train"

    def plan(self, world: World) -> None:
        if self.desire == "eat":
            self.target_tile = world.tiles[f"eat_{self.tile_prefix}"]
            self.state = "moving"
        elif self.desire == "rest":
            self.target_tile = world.tiles[f"rest_{self.tile_prefix}"]
            self.state = "moving"
        else:
            self.target_tile = world.training_pos
            if self.state != "queued":
                self.state = "moving"

    def act(self, world: World) -> str:
        # 1. If busy, continue or finish action
        if self.busy_ticks > 0:
            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                return self.finish_action()
            return f"{self.name} continues {self.current_action}."

        # 2. If queued, watch the training tile or leave due to new desire
        if self.state == "queued":
            # If we’re queued but desire changed (handled in reflect),
            # just fall through to normal movement with new target.
            if self.desire != "train":
                # state & target already updated in reflect/plan
                pass
            else:
                # Still want to train → check if training tile is free
                if world.is_tile_free(world.training_pos):
                    # Leave queue, go claim training tile
                    self.state = "moving"
                    self.target_tile = world.training_pos
                else:
                    return (
                        f"{self.name} stands aside, waiting for a turn "
                        "at the training grounds."
                    )

        # 3. Not busy, not (or no longer) queued → move or start action
        if self.target_tile is None:
            # No clear goal (shouldn’t really happen)
            self.state = "idle"
            return f"{self.name} stands still, uncertain."

        x, y = self.position
        tx, ty = self.target_tile

        # If at target tile
        if (x, y) == (tx, ty):
            if self.desire == "train":
                # Check for contention: if someone else is already training here,
                # we move into queue mode instead of starting
                if not world.is_tile_free(world.training_pos,
                                          ignore=self.name):
                    return self._enter_queue(world)
            return self.start_action()

        # If moving toward training tile and next step would collide,
        # enter queue instead of body-checking
        new_x, new_y = x, y
        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        next_pos = (new_x, new_y)

        if self.desire == "train" and next_pos == world.training_pos:
            # If training tile is occupied, queue instead of stepping in
            if not world.is_tile_free(world.training_pos, ignore=self.name):
                return self._enter_queue(world)

        # Normal move
        world.move_monk(self.name, next_pos)
        self.stamina = max(0, self.stamina - 1)
        self.state = "moving"

        direction = self._dir((x, y), (new_x, new_y))
        return f"{self.name}: {self.last_thought} I walk {direction}."

    # ----------------- QUEUE LOGIC ----------------- #

    def _enter_queue(self, world: World) -> str:
        self.state = "queued"
        self.waiting_tile = self._choose_waiting_tile(world)
        if self.waiting_tile is not None:
            # Move to waiting tile if not already there
            if self.position != self.waiting_tile:
                world.move_monk(self.name, self.waiting_tile)
            return (
                f"{self.name} steps aside and waits near the "
                "training grounds."
            )
        # No free waiting tile; just stay in place and wait
        return (
            f"{self.name} pauses. Another monk uses the "
            "training grounds. I will wait."
        )

    def _choose_waiting_tile(self, world: World) -> tuple[int, int] | None:
        tx, ty = world.training_pos
        candidates = [
            (tx - 1, ty),
            (tx + 1, ty),
            (tx, ty - 1),
            (tx, ty + 1),
        ]
        free_candidates = [
            c for c in candidates
            if world.is_inside(*c) and world.is_tile_free(c, ignore=self.name)
        ]
        return free_candidates[0] if free_candidates else None

    # ----------------- ACTIONS ----------------- #

    def start_action(self) -> str:
        self.state = "acting"

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

    def finish_action(self) -> str:
        if self.current_action == "eating":
            self.hunger = max(0, self.hunger - 30)
            self.stamina = min(100, self.stamina + 10)
            msg = f"{self.name} finishes eating and feels nourished."
        elif self.current_action == "resting":
            self.stamina = min(100, self.stamina + 40)
            self.hunger = min(100, self.hunger + 5)
            msg = f"{self.name} rises from rest with renewed strength."
        else:  # training
            self.skill += 5
            self.stamina = max(0, self.stamina - 20)
            self.hunger = min(100, self.hunger + 10)
            msg = f"{self.name} finishes training and feels sharper."

        self.current_action = None
        self.busy_ticks = 0
        self.state = "idle"
        self.target_tile = None
        self.waiting_tile = None
        return msg

    # ----------------- HELPERS ----------------- #

    def _dir(self, old: tuple[int, int],
             new: tuple[int, int]) -> str:
        ox, oy = old
        nx, ny = new
        if nx > ox:
            return "east"
        if nx < ox:
            return "west"
        if ny > oy:
            return "south"
        return "north"


# ----------------- SIMULATION ----------------- #

def game_loop(ticks: int = 50) -> None:
    world = World()

    personality_A = {
        "food_priority": 1.0,
        "rest_priority": 1.2,
        "training_drive": 0.8,
    }

    personality_B = {
        "food_priority": 0.8,
        "rest_priority": 0.7,
        "training_drive": 1.4,
    }

    monks = [
        Monk("Zhang Wei", personality_A, "A"),
        Monk("Zhang Li", personality_B, "B"),
    ]

    # Opposite corners for nice symmetry
    world.register_monk("Zhang Wei", (0, 0))
    world.register_monk("Zhang Li", (4, 4))

    # To give higher training_drive some priority in conflicts,
    # act higher-drive monks first each tick.
    monks_sorted = sorted(
        monks,
        key=lambda m: m.personality["training_drive"],
        reverse=True,
    )

    for tick in range(1, ticks + 1):
        print(f"\n==== TICK {tick} ====")

        for monk in monks_sorted:
            monk.observe(world)
            monk.update_memory()
            monk.reflect()
            monk.plan(world)
            log = monk.act(world)
            monk.hunger = min(100, monk.hunger + 1)
            print(log)

        # Uncomment to visualize the grid
        # world.print_world()
    
    # Show final stats after all ticks
    print("\n" + "=" * 50)
    print("FINAL STATS AFTER 50 TICKS")
    print("=" * 50)
    for monk in monks:
        print(f"\n{monk.name}:")
        print(f"  Hunger:   {monk.hunger}")
        print(f"  Stamina:  {monk.stamina}")
        print(f"  Skill:    {monk.skill}")


if __name__ == "__main__":
    game_loop()
