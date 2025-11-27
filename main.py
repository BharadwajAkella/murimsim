import random


class World:
    def __init__(self, width: int = 5, height: int = 5) -> None:
        self.width = width
        self.height = height

        self.training_pos = (2, 2)

        self.tiles = {
            "eat_A": (0, 4),
            "rest_A": (0, 3),
            "eat_B": (4, 0),
            "rest_B": (4, 1),
        }

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


class Monk:
    def __init__(self, name: str, personality: dict[str, float],
                 tile_prefix: str) -> None:
        self.name = name

        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        self.personality = personality
        self.tile_prefix = tile_prefix

        self.busy_ticks = 0
        self.current_action: str | None = None
        self.desire: str | None = None
        self.target_tile: tuple[int, int] | None = None

        self.state = "idle"  # idle / moving / queued / acting
        self.waiting_tile: tuple[int, int] | None = None

        self.memory = {
            "hunger_trend": [],
            "stamina_trend": [],
        }

        self.last_thought = ""
        self.position = (0, 0)

    # --------------------- Dialogue --------------------- #

    def speak(self, msg: str) -> None:
        print(f'{self.name} says: "{msg}"')

    def dialogue_for_desire(self, desire: str) -> str:
        fp = self.personality["food_priority"]
        rp = self.personality["rest_priority"]
        td = self.personality["training_drive"]

        if desire == "train":
            if td > 1.2:
                return "If I don’t push harder, I fall behind."
            return "This body strengthens only through pain."

        if desire == "eat":
            if fp > 1.1:
                return "Hunger is dragging me down. I need fuel."
            return "I should refill my strength."

        if desire == "rest":
            if rp > 1.1:
                return "A brief reset will let me push further."
            return "No point training on a failing body."

        return "I move."

    def dialogue_queue_enter(self) -> str:
        return "Tch. Someone got here first. I’ll wait my turn."

    def dialogue_action_finish(self, action: str) -> str:
        if action == "training":
            return "Good. I can feel myself sharpening."
        if action == "eating":
            return "Fuel restored. Back to the grind."
        if action == "resting":
            return "Breath steady. Time to push again."
        return "Done."

    # --------------------- Pipeline --------------------- #

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

        # If queued, maybe abandon
        if self.state == "queued":
            hunger_pressure = self.hunger * fp
            fatigue_pressure = (100 - self.stamina) * rp
            if hunger_pressure > 80 or fatigue_pressure > 70:
                self.desire = "eat" if hunger_pressure > fatigue_pressure else "rest"
                self.speak("No point waiting if my body is failing.")
                self.state = "idle"
                self.waiting_tile = None
                return

        hunger_pressure = self.hunger * fp
        fatigue_pressure = (100 - self.stamina) * rp

        if hunger_pressure > 70:
            self.desire = "eat"
        elif fatigue_pressure > 40:
            self.desire = "rest"
        else:
            self.desire = "train"

        # Dialogue on desire
        self.speak(self.dialogue_for_desire(self.desire))

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
        # Busy → continue
        if self.busy_ticks > 0:
            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                msg = self.finish_action()
                return msg
            return f"{self.name} continues {self.current_action}."

        # Queue logic
        if self.state == "queued":
            if world.is_tile_free(world.training_pos):
                self.state = "moving"
                self.target_tile = world.training_pos
            else:
                return f"{self.name} waits, watching the training tile."

        # Move or start action
        if self.target_tile is None:
            return f"{self.name} stands idle."

        x, y = self.position
        tx, ty = self.target_tile

        if (x, y) == (tx, ty):
            if self.desire == "train" and not world.is_tile_free(
                world.training_pos, ignore=self.name
            ):
                return self._enter_queue(world)
            return self.start_action()

        # move toward target
        new_x, new_y = x, y
        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        next_pos = (new_x, new_y)

        if self.desire == "train" and next_pos == world.training_pos:
            if not world.is_tile_free(world.training_pos, ignore=self.name):
                return self._enter_queue(world)

        world.move_monk(self.name, next_pos)
        self.stamina = max(0, self.stamina - 1)

        direction = self._dir((x, y), (new_x, new_y))
        return f"{self.name}: I walk {direction}."

    # ----------------- Queue ----------------- #

    def _enter_queue(self, world: World) -> str:
        self.state = "queued"
        self.speak(self.dialogue_queue_enter())

        tx, ty = world.training_pos
        candidates = [
            (tx - 1, ty),
            (tx + 1, ty),
            (tx, ty - 1),
            (tx, ty + 1),
        ]

        for c in candidates:
            if world.is_inside(*c) and world.is_tile_free(c, ignore=self.name):
                self.waiting_tile = c
                world.move_monk(self.name, c)
                return f"{self.name} steps aside."

        return f"{self.name} waits in place."

    # ----------------- Actions ----------------- #

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
            self.hunger -= 30
            self.hunger = max(0, self.hunger)
            self.stamina = min(100, self.stamina + 10)
        elif self.current_action == "resting":
            self.stamina = min(100, self.stamina + 40)
            self.hunger = min(100, self.hunger + 5)
        else:
            self.skill += 5
            self.stamina = max(0, self.stamina - 20)
            self.hunger = min(100, self.hunger + 10)

        self.speak(self.dialogue_action_finish(self.current_action))

        msg = f"{self.name} finishes {self.current_action}."
        self.current_action = None
        self.busy_ticks = 0
        self.state = "idle"
        self.target_tile = None
        self.waiting_tile = None

        return msg

    # ----------------- Helpers ----------------- #

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


# ------------------ Simulation ------------------ #

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

    world.register_monk("Zhang Wei", (0, 0))
    world.register_monk("Zhang Li", (4, 4))

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
            print(monk.act(world))
        # world.print_world()

    print("\n==== FINAL STATS ====")
    for monk in monks_sorted:
        print(
            f"{monk.name}: Hunger={monk.hunger}, "
            f"Stamina={monk.stamina}, Skill={monk.skill}, State={monk.state}"
        )


if __name__ == "__main__":
    game_loop()
