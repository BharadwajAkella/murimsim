import random


# ============================================================
#                       WORLD MODEL
# ============================================================

class World:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

        # Shared training tile
        self.training_pos = (2, 2)

        # Private tiles for basic needs
        self.tiles = {
            "eat_A": (0, 4),
            "rest_A": (0, 3),
            "eat_B": (4, 0),
            "rest_B": (4, 1),
        }

        self.monk_positions = {}

    def is_inside(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def register_monk(self, name, pos):
        self.monk_positions[name] = pos

    def move_monk(self, name, new_pos):
        x, y = new_pos
        if not self.is_inside(x, y):
            raise ValueError(f"{name} attempted off-grid: {new_pos}")
        self.monk_positions[name] = (x, y)

    def is_tile_free(self, pos, ignore=None):
        for name, p in self.monk_positions.items():
            if ignore and name == ignore:
                continue
            if p == pos:
                return False
        return True


# ============================================================
#                      MONK / AGENT
# ============================================================

class Monk:
    def __init__(self, name, personality, tile_prefix):
        self.name = name

        # Core stats
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        # NEW FOR M5
        self.hp = 100
        self.injury = 0

        # Personality modifiers
        self.personality = personality
        self.tile_prefix = tile_prefix

        # State & action management
        self.state = "idle"     # idle, moving, acting, queued, combat, retreat
        self.desire = None
        self.current_action = None
        self.busy_ticks = 0
        self.target_tile = None
        self.waiting_tile = None

        self.position = (0, 0)
        self.last_thought = ""

    # ============================================================
    #                       DIALOGUE
    # ============================================================

    def speak(self, msg):
        print(f'{self.name} says: "{msg}"')

    def speak_desire(self):
        d = self.desire
        td = self.personality["training_drive"]

        if d == "train":
            if td > 1.2:
                return "If I don’t push harder, I fall behind."
            return "This body sharpens only through pain."

        if d == "eat":
            return "I need fuel if I want to keep up."

        if d == "rest":
            return "A quick reset, then I grind again."

        return "I move."

    def speak_combat_start(self):
        return "If you want the spot, take it from me."

    def speak_combat_hit(self):
        return random.choice([
            "Endure this.",
            "Too slow.",
            "Push harder.",
        ])

    def speak_combat_loss(self):
        return "Tch… I’ll repay this soon."

    def speak_combat_win(self):
        return "Move. I’m taking this today."

    # ============================================================
    #                       PIPELINE
    # ============================================================

    def observe(self, world):
        self.position = world.monk_positions[self.name]

    def reflect(self):
        """Set desire based on simple weighted needs."""
        if self.state == "retreat":
            self.desire = "rest"
            return

        fp = self.personality["food_priority"]
        rp = self.personality["rest_priority"]

        hunger_pressure = self.hunger * fp
        fatigue_pressure = (100 - self.stamina) * rp

        if hunger_pressure > 70:
            self.desire = "eat"
        elif fatigue_pressure > 40:
            self.desire = "rest"
        else:
            self.desire = "train"

        self.speak(self.speak_desire())

    def plan(self, world):
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

    def act(self, world, other):
        # ==========================================
        # HANDLE RETREAT
        # ==========================================
        if self.state == "retreat":
            return self._retreat_step(world)

        # ==========================================
        # HANDLE ACTION IN PROGRESS
        # ==========================================
        if self.busy_ticks > 0:
            self.busy_ticks -= 1
            if self.busy_ticks == 0:
                return self.finish_action()
            return f"{self.name} continues {self.current_action}."

        # ==========================================
        # COMBAT CHECK (ONLY TRIGGER CASE)
        # ==========================================
        if self._combat_should_start(world, other):
            return self._enter_combat(other)

        if self.state == "combat":
            return self._combat_step(other)

        # ==========================================
        # QUEUE BEHAVIOR
        # ==========================================
        if self.state == "queued":
            if world.is_tile_free(world.training_pos):
                self.state = "moving"
                self.target_tile = world.training_pos
            else:
                return f"{self.name} waits for the training tile."

        # ==========================================
        # MOVEMENT OR START ACTION
        # ==========================================
        if self.target_tile is None:
            return f"{self.name} stands idle."

        x, y = self.position
        tx, ty = self.target_tile

        # At target tile
        if (x, y) == (tx, ty):
            if self.desire == "train" and not world.is_tile_free(
                world.training_pos, ignore=self.name
            ):
                return self._enter_queue(world)

            return self.start_action()

        # Move toward target
        new_x, new_y = x, y
        if x != tx:
            new_x += 1 if tx > x else -1
        else:
            new_y += 1 if ty > y else -1

        world.move_monk(self.name, (new_x, new_y))
        self.stamina = max(0, self.stamina - 1)
        return f"{self.name} walks."

    # ============================================================
    #               QUEUE + RETREAT + ACTIONS
    # ============================================================

    def _enter_queue(self, world):
        self.state = "queued"
        return f"{self.name} steps aside to wait."

    def _retreat_step(self, world):
        # Move to rest tile
        rt = world.tiles[f"rest_{self.tile_prefix}"]
        if self.position != rt:
            x, y = self.position
            tx, ty = rt
            new_x = x + (1 if tx > x else -1 if tx < x else 0)
            new_y = y + (1 if ty > y else -1 if ty < y else 0)
            world.move_monk(self.name, (new_x, new_y))
            return f"{self.name} limps toward rest."

        # At rest tile → recover
        self.stamina = min(100, self.stamina + 20)
        self.hp = min(100, self.hp + 15)
        if self.hp > 60:
            self.state = "idle"
            return f"{self.name} recovers enough to return."
        return f"{self.name} rests to recover."

    def start_action(self):
        self.state = "acting"

        if self.desire == "eat":
            self.current_action = "eating"
            self.busy_ticks = 3
        elif self.desire == "rest":
            self.current_action = "resting"
            self.busy_ticks = 5
        else:
            self.current_action = "training"
            self.busy_ticks = 4

        return f"{self.name} begins {self.current_action}."

    def finish_action(self):
        action = self.current_action

        if action == "eating":
            self.hunger = max(0, self.hunger - 30)
            self.stamina = min(100, self.stamina + 10)

        elif action == "resting":
            self.stamina = min(100, self.stamina + 40)
            self.hunger = min(100, self.hunger + 5)

        else:  # training
            self.skill += 5
            self.hunger = min(100, self.hunger + 10)
            self.stamina = max(0, self.stamina - 20)

        self.state = "idle"
        self.current_action = None
        self.busy_ticks = 0
        self.target_tile = None
        return f"{self.name} finishes {action}."

    # ============================================================
    #                        COMBAT
    # ============================================================

    def _combat_should_start(self, world, other):
        if self.desire != "train":
            return False
        if other.desire != "train":
            return False

        # Both reached training tile at same time
        return (
            self.position == world.training_pos
            and other.position == world.training_pos
        )

    def _enter_combat(self, other):
        self.state = "combat"
        other.state = "combat"
        self.speak(self.speak_combat_start())
        return f"{self.name} enters combat!"

    def _combat_step(self, other):
        # Minimal 10-line combat resolution:
        atk = 5 + int(self.personality["training_drive"] * 3)
        dfs = 3 + int(self.personality["rest_priority"] * 2)
        dmg = max(1, atk + random.randint(0, 4) - (dfs + random.randint(0, 3)))
        other.hp -= dmg

        self.speak(self.speak_combat_hit())

        if other.hp <= 30:
            # Victory
            self.speak(self.speak_combat_win())
            other.speak(self.speak_combat_loss())
            other.state = "retreat"
            self.state = "idle"
            return f"{self.name} wins the clash!"

        return f"{self.name} strikes {other.name} for {dmg} damage."


# ============================================================
#                      SIMULATION LOOP
# ============================================================

def game_loop(ticks=40):
    world = World()

    monks = [
        Monk("Zhang Wei",
             {"food_priority": 1.0, "rest_priority": 1.2,
              "training_drive": 0.8},
             "A"),
        Monk("Zhang Li",
             {"food_priority": 0.8, "rest_priority": 0.7,
              "training_drive": 1.4},
             "B"),
    ]

    # Start positions
    world.register_monk("Zhang Wei", (0, 0))
    world.register_monk("Zhang Li", (4, 4))

    for tick in range(1, ticks + 1):
        print(f"\n==== TICK {tick} ====")

        # Deterministic ordering: higher training_drive acts first
        ordered = sorted(
            monks, key=lambda m: m.personality["training_drive"],
            reverse=True
        )

        # Pass the other monk into act() so combat can check both
        ordered[0].observe(world)
        ordered[1].observe(world)

        ordered[0].reflect()
        ordered[1].reflect()

        ordered[0].plan(world)
        ordered[1].plan(world)

        # act with reference to the other monk
        print(ordered[0].act(world, ordered[1]))
        print(ordered[1].act(world, ordered[0]))

        # global passive hunger tick
        for m in monks:
            m.hunger = min(100, m.hunger + 1)

    print("\n==== FINAL STATS ====")
    for m in monks:
        print(f"{m.name}: HP={m.hp}, Stamina={m.stamina}, "
              f"Hunger={m.hunger}, Skill={m.skill}, State={m.state}")


if __name__ == "__main__":
    game_loop()
