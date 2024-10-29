```Each plot Grid = {
                "planted": False,
                "crop_type": 0,
                "crop_state": "",
                "days": 0,
                "disease": False
            }
```
```
seed types = [1, 2, 3]
```
```
SeederAgent = {
    self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'collect_seeds', 'plant_seeds', 'drop_seeds']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_seeds = False
        self.seeds = 0
        self.seed_type = 0 # 0 - None, 1 - Type 1, 2 - Type 2, 3 - Type 3
        self.capacity = 4
}
```
```
WaterAgent = {
    self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'collect_water', 'water_crops', 'drop_water']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_water = False
        self.water_units = 0
        self.capacity = 4
}
```
```
HarvesterAgent = {
    self.actions = ['move_up', 'move_down', 'move_left', 'move_right', 'idle', 'rotate_clockwise', 'rotate_anticlockwise', 'harvest_crops', 'drop_crops']
        self.action_space = spaces.Discrete(8)
        self.pos = pos
        self.facing = 2 # 0 - Up, 1 - Right, 2 - Down, 3 - Left
        self.holding_crops = False
        self.crop_units = 0
        self.crop_type = 0 # 0 - None, 1 - Type 1, 2 - Type 2, 3 - Type 3
        self.crop_state = ""
}
```

```
crop_states = ["planted", "sapling","harvest", "dried", "dead"]
```
days crop_state
0    planted
5    sapling
25   harvest
35   dried
40   dead