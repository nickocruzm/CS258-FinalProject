# CS258-FinalProject

## Pong

- `frameskip=4`
- `repeat_action_probability=0.25`
- `full_action_space=True`
- `mode = 0` (default)
- Difficult = 0 (default)
- 
### Action Space
0: NOOP
1: FIRE
2: RIGHT
3: LEFT
4: RIGHTFIRE
5: LEFTFIRE

### Observation Space
#### Pixel Dimensions:

- 210-pixel height
- 160-pixel width
- 3 color channels (RGB)


##### Notes

```python3
    gym.Env.render(self) 
```

Compute the render frames as specified by render_mode attribute during initialization of the environment.

The set of supported modes varies per environment. (And some third-party environments may not support rendering at all.) By convention, if render_mode is:

None (default): no render is computed.

human: render return None. The environment is continuously rendered in the current display or terminal. Usually for human consumption.

rgb_array: return a single frame representing the current state of the environment. A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.

rgb_array_list: return a list of frames representing the states of the environment since the last reset. Each frame is a numpy.ndarray with shape (x, y, 3), as with rgb_array.

ansi: Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
