# Murim Simulation

A tick-based agent simulation where monks navigate a shared world, managing hunger, stamina, and skill through training, eating, and resting.

## Architecture

### M1: Single Monk (Commit: e6fac3f)
- Single autonomous monk agent
- Observe → Reflect → Plan → Act brain loop
- Memory module with 5-tick trend tracking
- Adaptive behavior based on hunger/stamina trends
- Success/failure streak tracking

### M1.5: Multi-Monk with Conflicts (Commit: e3f90fb)
- Multiple monks in shared world
- Tile occupancy tracking
- Path blocking detection
- Resource conflict handling
- Per-monk conflict counters

## Usage

```python
# Run single monk simulation (M1)
python3 main.py  # Edit line: game_loop(ticks=50, num_monks=1)

# Run multi-monk simulation (M1.5)
python3 main.py  # Edit line: game_loop(ticks=50, num_monks=2)
```

## Git History

```bash
# View commits
git log --oneline

# Compare M1 vs M1.5
git diff e6fac3f e3f90fb

# Checkout M1 version
git checkout e6fac3f
```

## Key Features

- **Memory System**: Tracks hunger/stamina trends over last 5 ticks
- **Adaptive Reflection**: Adjusts behavior based on detected trends
- **Conflict Detection**: Monks wait when resources are occupied
- **Clean Architecture**: World manages positions, Monks manage behavior
