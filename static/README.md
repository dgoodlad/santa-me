# Santa Hat Configuration

## santa_hat.json

Semantic positioning configuration defines how the hat should be sized and positioned relative to facial features:

**Parameters:**
- **width_reference**: Which facial measurement to use (`eye_distance` or `forehead_width`)
- **width_multiplier**: How many times the reference measurement for hat width
- **hat_anchor_point**: Which point on the hat image (normalized 0-1 coordinates) aligns with the target
  - `x`: 0=left, 0.5=center, 1=right
  - `y`: 0=top, 0.5=middle, 1=bottom
- **horizontal_center**: Horizontal target position (`midpoint_between_eyes` or `forehead_top`)
- **vertical_anchor**: Vertical reference point (`forehead_top`)
- **vertical_offset_px**: Pixels to offset vertically (positive = move down, negative = move up)

This semantic approach makes hat positioning consistent across different head sizes and handles tilted heads correctly.
