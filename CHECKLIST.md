# Shadow Boxer AI - Checklist for Next Steps

## 1. Hình ảnh và giao diện của game (Art & UI)
- [x] Cơ bản: Pygame rendering với arena, opponent, gloves, HUD
- [ ] Thay thế rectangles bằng 2D sprites (opponent states, gloves, background)
- [ ] Thêm layering Z-index (background, opponent, gloves, HUD, VFX)
- [ ] Thiết kế giao diện HUD đầy đủ (health bars, timer, round info)
- [ ] Thêm VFX (hit sparks, blood effects)
- [ ] Menu chính, pause screen, game over screen

## 2. Frame của từng hành động trong game (Frames for Actions)
- [x] Cơ bản: Pose tracking với MediaPipe
- [ ] Thêm state machine cho opponent (IDLE, BLOCK, ATTACK, HURT)
- [ ] Animation frames cho từng state (sprite sheets)
- [ ] Transition giữa states (timers, conditions)
- [ ] Player blocking detection (hands raised)
- [ ] Stun effects khi trúng đòn

## 3. Hitbox của game (Hitboxes)
- [x] Cơ bản: Head và Body hitboxes cho opponent
- [ ] Tinh chỉnh kích thước và vị trí hitboxes
- [ ] Thêm hitboxes cho player (để opponent attack)
- [ ] Debug mode: hiển thị hitboxes (đã có, có thể toggle)
- [ ] Collision detection cải tiến (pixel-perfect nếu dùng sprites)

## 4. Máu của đối thủ (Opponent's Health)
- [x] Cơ bản: Health bar cho opponent
- [ ] Thêm visual feedback khi mất máu (flash, shake)
- [ ] Regenerate hoặc heal mechanics (nếu cần)
- [ ] Multiple opponents hoặc boss fights

## 5. Dame của người chơi và đối thủ (Damage for Player & Opponent)
- [x] Cơ bản: Damage calculation (head: 12, body: 6)
- [ ] Thêm damage types (critical, normal, combo bonuses)
- [ ] Player health và damage từ opponent attacks
- [ ] Balancing damage values
- [ ] Sound effects cho hits

## 6. Các hoạt động trong game (Game Activities: Levels, Shop, Tournaments)
- [x] Cơ bản: Single fight mode
- [ ] Thêm multiple rounds per match
- [ ] Level progression (unlock opponents, arenas)
- [ ] Shop system (buy gloves, upgrades)
- [ ] Tournament mode (bracket fights)
- [ ] Leaderboards và achievements
- [ ] Save/load progress

## Additional Enhancements
- [ ] Sound & Music (background music, hit sounds, crowd noise)
- [ ] Multiplayer (local or online)
- [ ] Difficulty settings
- [ ] Customization (glove colors, arena themes)
- [ ] Performance optimization (lower FPS if needed)
- [ ] Cross-platform testing (Windows/macOS/Linux)

## Current Status
- Prototype core loop: ✅ Working
- Pose tracking & punch detection: ✅ Working
- Basic rendering & collision: ✅ Working
- Debug webcam overlay: ✅ Added

Next priority: Implement opponent state machine and replace basic shapes with sprites.</content>
<parameter name="filePath">d:\PROJECTGAME\CHECKLIST.md