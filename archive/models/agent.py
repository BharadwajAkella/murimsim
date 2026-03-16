
class AgentState:
    IDLE = 'idle'
    MOVING = 'moving'
    WORKING = 'working'
    ACTING = 'acting'
    QUEUED = 'queued'
    COMBAT = 'combat'
    RETREAT = 'retreat'

class Agent:
    def __init__(self, name, personality, tile_prefix):
        self.name = name
        self.personality = personality
        self.tile_prefix = tile_prefix

        # Core stats
        self.hunger = 0
        self.stamina = 100
        self.skill = 0

        self.hp = 100
        self.injury = 0

        # State management
        self.state = AgentState.IDLE
        self.desire = None
        self.current_action = None
        self.busy_ticks = 0
        self.target_tile = None
        self.waiting_tile = None

        self.position = (0, 0)  # (x, y) coordinates
        self.last_thought = ""

    # Dialogue
    def speak(self, message):
        print(f"{self.name} says: {message}")

    def speak_desire(self):
        desire = self.desire
        training_drive = self.personality.get('training_drive', 0)
        if desire == 'train':
            if training_drive > 1.2:
                return "I must train harder!"
            elif training_drive > 0.8:
                return "I should do some training."
            else:
                return "Training is not my priority right now."
        elif desire == 'eat':
            return "I need to find some food."
        elif desire == 'rest':
            return "I should take a break."
        return "I need to move"
    
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
    

    def observe(self):
        # Observe the world 2 tiles around the agent
        observed_tiles = []
        x, y = self.position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                observed_tiles.append((x + dx, y + dy))