import random

class World:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        # For M2 → each monk gets private tiles (parallel lives)
        self.tiles = {
            "train_A": (2, 0),
            "rest_A": (0, 4),
            "eat_A":  (2, 4),

            "train_B": (2, 4),
            "rest_B":  (4, 0),
            "eat_B":   (2, 0),
        }

        self.monk_positions = {}

    def register_monk(self, name, pos):
        self.monk_positions[name] = pos

    def move_monk(self, name, new_pos):
        x, y = new_pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} attempted to move off-grid: {new_pos}")
        self.monk_positions[name] = (x, y)

    def print_world(self):
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        # Mark monks
        for name, (mx, my) in self.monk_positions.items():
            grid[my][mx] = " M"

        for row in grid:
            print("".join(row))


class Monk:
    def __init__(self, name, personality, tile_prefix):
        self.name = name

        # Core stats
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        # Action state
        self.busy_ticks = 0
        self.current_action = None
        self.desire = None
        self.target_tile = None
        self.tile_prefix = tile_prefix

        # Personality weights
        self.personality = personality

        # Memory (simple trends)
        self.memory = {
            "hunger_trend": [],
            "stamina_trend": []
        }

        self.last_thought = ""
        self.position = (0, 0)

    # ---------------- PIPELINE ---------------- #

    def observe(self, world):
        self.position = world.monk_positions[self.name]

    def update_memory(self):
        # Track last 5 hunger/stamina values
        self.memory["hunger_trend"].append(self.hunger)
        if len(self.memory["hunger_trend"]) > 5:
            self.memory["hunger_trend"].pop(0)

        self.memory["stamina_trend"].append(self.stamina)
        if len(self.memory["stamina_trend"]) > 5:
            self.memory["stamina_trend"].pop(0)

    def reflect(self):
        """
        M2 logic: use personality weights to shape desires.
        """
        fp = self.personality["food_priority"]
        rp = self.personality["rest_priority"]
        td = self.personality["training_drive"]

        # Weighted hunger urgency
        if self.hunger * fp > 70:
            self.last_thought = f"I should eat before hunger distracts me."
            self.desire = "eat"
            return "eat"

        # Weighted fatigue sensitivity
        if (100 - self.stamina) * rp > 40:
            self.last_thought = f"My body seeks rest."
            self.desire = "rest"
            return "rest"

        # Weighted preference for training
        self.last_thought = f"Training sharpens the mind and body."
        self.desire = "train"
        return "train"

    def plan(self, world):
        prefix = self.tile_prefix

        if self.desire == "eat":
            self.target_tile = world.tiles[f"eat_{prefix}"]
        elif self.desire == "rest":
            self.target_tile = world.tiles[f"rest_{prefix}"]
        else:
            self.target_tile = world.tiles[f"train_{prefix}"]

    def act(self, world):
        # Continue current action
        if self.busy_ticks > 0:
            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                return self.finish_action()
            return f"{self.name} continues {self.current_action}."

        # Not busy → move or start action
        x, y = self.position
        tx, ty = self.target_tile

        # At destination → start action
        if (x, y) == (tx, ty):
            return self.start_action()

        # Move toward target (taxicab)
        new_x, new_y = x, y
        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        world.move_monk(self.name, (new_x, new_y))
        self.stamina = max(0, self.stamina - 1)

        direction = self._dir((x, y), (new_x, new_y))
        return f"{self.name}: {self.last_thought} I walk {direction}."

    # ---------------- ACTIONS ---------------- #

    def start_action(self):
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
            msg = f"{self.name} wakes refreshed."

        else:  # training
            self.skill += 5
            self.stamina = max(0, self.stamina - 20)
            self.hunger = min(100, self.hunger + 10)
            msg = f"{self.name} finishes training."

        self.current_action = None
        self.target_tile = None
        return msg

    # ---------------- HELPERS ---------------- #

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

    # PERSONALITIES
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

    for tick in range(1, ticks + 1):
        print(f"\n==== TICK {tick} ====")

        for monk in monks:
            monk.observe(world)
            monk.update_memory()
            monk.reflect()
            monk.plan(world)
            log = monk.act(world)
            monk.hunger = min(100, monk.hunger + 1)
            print(log)

        # Uncomment to visualize
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
