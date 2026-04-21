import os
import sys
import time
import math
import random
import cv2
import numpy as np
import mediapipe as mp
import pygame
from pathlib import Path

# Shadow Boxer AI (FPP boxing) prototype
# - OpenCV webcam capture
# - MediaPipe Pose landmark tracking
# - Pygame rendering of arena/opponent/gloves/hud

# WIDTH, HEIGHT = 1280, 720
FPS = 30

# # Opponent hitbox partial zones in portrait coordinate space
# OPP_X, OPP_Y = WIDTH // 2, HEIGHT // 2
# OPP_W, OPP_H = 330, 570
# HEAD_BOX = pygame.Rect(OPP_X - 60, OPP_Y - 240, 120, 120)
# BODY_BOX = pygame.Rect(OPP_X - 105, OPP_Y - 120, 210, 270)

# gameplay parameters
# Giảm ngưỡng để nhận diện cú đấm nhạy hơn
PUNCH_SPEED_THRESHOLD = 10.0  # px/frame velocity
PUNCH_DEPTH_THRESHOLD = 0.03  # normalized Z movement toward camera
PUNCH_ANGLE_THRESHOLD = 130.0  # degrees shoulder-elbow-wrist straightness
HIT_COOLDOWN = 0.3  # seconds after registered hit
ENEMY_ATTACK_INTERVAL_MIN = 2.0
ENEMY_ATTACK_INTERVAL_MAX = 4.5
ENEMY_BLOCK_CHANCE = 0.35
ENEMY_ATTACK_DURATION = 0.7
ENEMY_BLOCK_DURATION = 2.5


def normalize_point(landmark, frame_w, frame_h):
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(a, b, c):
    # angle at b
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    dot = np.dot(ab, cb)
    mag = np.linalg.norm(ab) * np.linalg.norm(cb)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def is_blocking_pose(landmarks, frame_w, frame_h):
    left_wrist = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST], frame_w, frame_h)
    right_wrist = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST], frame_w, frame_h)
    left_elbow = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW], frame_w, frame_h)
    right_elbow = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW], frame_w, frame_h)
    left_shoulder = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER], frame_w, frame_h)
    right_shoulder = normalize_point(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER], frame_w, frame_h)

    head_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                   min(left_shoulder[1], right_shoulder[1]) - 80)
    chest_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)
    block_distance = 150

    left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)

    left_guard = (
        (left_wrist[1] < chest_center[1] + 60 and distance(left_wrist, head_center) < block_distance)
        and left_elbow_angle < 130
    )
    right_guard = (
        (right_wrist[1] < chest_center[1] + 60 and distance(right_wrist, head_center) < block_distance)
        and right_elbow_angle < 130
    )

    # nếu tay gần ngực và khuỷu gập, xem như đang che chắn
    return left_guard or right_guard


def map_to_screen(point, frame_w, frame_h):
    x, y = point
    # already mirrored in camera space, map directly to game window
    return int(x / frame_w * WIDTH), int(y / frame_h * HEIGHT)


def load_image(filename, size=None):
    """Load image from IMGS folder"""
    img_path = os.path.join(os.path.dirname(__file__), 'IMGS', filename)
    if not os.path.exists(img_path):
        print(f'Warning: Image not found: {img_path}')
        return None
    
    image = pygame.image.load(img_path)
    image = image.convert_alpha()
    
    if size:
        image = pygame.transform.scale(image, size)
    
    return image


def load_glove_image(filename, size=52):
    """Load glove image and process alpha for green background"""
    img_path = os.path.join(os.path.dirname(__file__), 'IMGS', filename)
    if not os.path.exists(img_path):
        print(f'Warning: Glove image not found: {img_path}')
        return None
    
    # Load image and convert to per-pixel alpha
    glove = pygame.image.load(img_path)
    glove = glove.convert_alpha()
    
    # Get pixel array and convert green (0, 255, 0) to transparent
    pixelarray = pygame.PixelArray(glove)
    green_color = pygame.Color(0, 255, 0)
    
    # Iterate through pixels and make green transparent
    for i in range(glove.get_width()):
        for j in range(glove.get_height()):
            pixel = glove.get_at((i, j))
            # Check if pixel is close to pure green
            if pixel[0] < 50 and pixel[1] > 200 and pixel[2] < 50:
                glove.set_at((i, j), (pixel[0], pixel[1], pixel[2], 0))
    
    # Scale to desired size
    glove = pygame.transform.scale(glove, (size, size))
    return glove


def load_audio(filename):
    """Load audio file from Music folder - fallback to .wav if original fails"""
    audio_path = os.path.join(os.path.dirname(__file__), 'Music', filename)
    
    # Try original file first
    if os.path.exists(audio_path):
        try:
            sound = pygame.mixer.Sound(audio_path)
            print(f'Successfully loaded audio: {filename}')
            return sound
        except Exception as e:
            print(f'Info: Cannot load {filename} ({e}), attempting .wav alternative...')
            
            # If MP4 fails, try WAV version
            if filename.endswith('.mp4'):
                wav_filename = filename.replace('.mp4', '.wav')
                wav_path = os.path.join(os.path.dirname(__file__), 'Music', wav_filename)
                if os.path.exists(wav_path):
                    try:
                        sound = pygame.mixer.Sound(wav_path)
                        print(f'Successfully loaded audio: {wav_filename}')
                        return sound
                    except Exception as e2:
                        print(f'Error loading {wav_filename}: {e2}')
    else:
        print(f'Warning: Audio file not found: {audio_path}')
    
    return None


def load_music(filename):
    """Load music file from Music folder - tries .wav if .mp4 fails"""
    music_path = os.path.join(os.path.dirname(__file__), 'Music', filename)
    
    # Try original file first
    if os.path.exists(music_path):
        try:
            pygame.mixer.music.load(music_path)
            print(f'Successfully loaded music: {filename}')
            return True
        except Exception as e:
            print(f'Info: Cannot load {filename} ({e}), attempting .wav alternative...')
            
            # If MP4 fails, try WAV version
            if filename.endswith('.mp4'):
                wav_filename = filename.replace('.mp4', '.wav')
                wav_path = os.path.join(os.path.dirname(__file__), 'Music', wav_filename)
                if os.path.exists(wav_path):
                    try:
                        pygame.mixer.music.load(wav_path)
                        print(f'Successfully loaded: {wav_filename}')
                        return True
                    except Exception as e2:
                        print(f'Error loading {wav_filename}: {e2}')
    else:
        print(f'Warning: Music file not found: {music_path}')
    
    return False


def draw_glove_at_position(screen, glove_image, position):
    """Draw glove image centered at position"""
    if glove_image is None:
        return
    rect = glove_image.get_rect(center=position)
    screen.blit(glove_image, rect)


def draw_menu(screen, selected_index, background):
    """Draw a gritty, underground-style main menu"""
    screen.blit(background, (0, 0))
    
    # Lớp phủ tối màu (Dark/Blood overlay) để làm nổi bật chữ
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((15, 5, 5, 210)) # Màu đen hơi ám đỏ
    screen.blit(overlay, (0, 0))

    # Dùng các font Unicode ổn định cho tiếng Việt
    title_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 110, bold=True)
    menu_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 55, bold=True)
    hint_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 24, bold=True)

    # Vẽ Title với hiệu ứng "Bóng đổ / Xịt sơn" (Graffiti shadow)
    title_text = 'SHADOW BOXER'
    title_shadow = title_font.render(title_text, True, (100, 10, 10)) # Đỏ máu thẫm
    title_surf = title_font.render(title_text, True, (220, 220, 220)) # Trắng xám
    
    title_x = WIDTH // 2 - title_surf.get_width() // 2
    title_y = HEIGHT // 4
    
    # Vẽ bóng trước (lệch xuống và sang phải), rồi vẽ chữ chính đè lên
    screen.blit(title_shadow, (title_x + 6, title_y + 6))
    screen.blit(title_surf, (title_x, title_y))

    # Đường viền phân cách gồ ghề (Caution line)
    pygame.draw.rect(screen, (150, 30, 30), (WIDTH // 2 - 250, title_y + 130, 500, 4))
    pygame.draw.rect(screen, (80, 10, 10), (WIDTH // 2 - 250, title_y + 134, 500, 2))

    menu_items = ['ENTER THE RING', 'COWARD\'S ESCAPE (QUIT)']
    for index, text in enumerate(menu_items):
        if index == selected_index:
            # Item đang được chọn: Màu vàng rỉ sét, có mũi tên
            display_text = f"X  {text}  X"
            color = (255, 180, 50) 
        else:
            # Item không được chọn: Màu xám tro
            display_text = text
            color = (120, 120, 120)

        item_surf = menu_font.render(display_text, True, color)
        item_x = WIDTH // 2 - item_surf.get_width() // 2
        item_y = HEIGHT // 2 + 50 + index * 80
        
        # Vẽ bóng cho menu
        shadow_surf = menu_font.render(display_text, True, (0, 0, 0))
        screen.blit(shadow_surf, (item_x + 3, item_y + 3))
        screen.blit(item_surf, (item_x, item_y))

    # Hint text ở dưới cùng
    hint_surf = hint_font.render('USE [UP]/[DOWN] TO NAVIGATE - [ENTER] TO ACCEPT', True, (150, 30, 30))
    screen.blit(hint_surf, (WIDTH // 2 - hint_surf.get_width() // 2, HEIGHT - 80))
    pygame.display.flip()


def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current = ''
    for word in words:
        test_line = f'{current} {word}' if current else word
        if font.size(test_line)[0] > max_width:
            if current:
                lines.append(current)
            current = word
        else:
            current = test_line
    if current:
        lines.append(current)
    return lines


def get_unicode_font(names, size, bold=False, italic=False):
    if isinstance(names, str):
        names = [names]

    safe_fonts = ['segoeui', 'arial', 'verdana', 'tahoma', 'calibri', 'timesnewroman', 'dejavusans']
    tried = []

    for name in safe_fonts + names:
        lower_name = name.lower()
        if lower_name in tried:
            continue
        tried.append(lower_name)
        try:
            font_path = pygame.font.match_font(name, bold=bold, italic=italic)
        except Exception:
            font_path = None
        if font_path:
            try:
                return pygame.font.Font(font_path, size)
            except Exception:
                pass

    try:
        return pygame.font.SysFont(names[0] if names else None, size, bold=bold, italic=italic)
    except Exception:
        return pygame.font.Font(None, size)


def draw_dialogue_box(screen, speaker, text):
    screen.fill((15, 15, 15))
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, (0, 0))

    box_rect = pygame.Rect(60, HEIGHT - 260, WIDTH - 120, 210)
    pygame.draw.rect(screen, (20, 20, 20, 220), box_rect, border_radius=12)
    pygame.draw.rect(screen, (220, 60, 60), box_rect, 4, border_radius=12)

    speaker_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 38, bold=True)
    body_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma', 'dejavusans'], 30)
    hint_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 22, bold=True)

    speaker_text = speaker_font.render(f'{speaker}:', True, (255, 220, 150))
    screen.blit(speaker_text, (box_rect.left + 20, box_rect.top + 18))

    lines = wrap_text(text, body_font, box_rect.width - 40)
    for i, line in enumerate(lines):
        line_surf = body_font.render(line, True, (230, 230, 230))
        screen.blit(line_surf, (box_rect.left + 20, box_rect.top + 70 + i * 34))

    hint_surf = hint_font.render('PRESS [ENTER] TO CONTINUE - [ESC] TO SKIP', True, (200, 200, 200))
    screen.blit(hint_surf, (box_rect.left + 20, box_rect.bottom - 36))
    pygame.display.flip()


def play_intro_video(screen, clock, filename):
    video_path = os.path.join(os.path.dirname(__file__), 'Video', filename)
    if not os.path.exists(video_path):
        print(f'Intro video not found: {video_path}')
        return

    _play_intro_video_silent(screen, clock, video_path)


def _play_intro_video_silent(screen, clock, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open intro video: {video_path}')
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_surface = pygame.image.frombuffer(frame.tobytes(), (WIDTH, HEIGHT), 'RGB')
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                return

        clock.tick(FPS)

    cap.release()


def run_intro_dialogue(screen, clock):
    dialogue = [
        ('SCAB', 'Lại mơ thấy cái đêm chết tiệt đó à? Tay mày vừa đấm vào không khí kìa. Cẩn thận đấy, bọn chó gác ngục mà thấy, chúng nó lại tưởng mày lên cơn dại.'),
        ('KID', '(Ngồi dậy, thở dốc) Không phải việc của ông, Scab.'),
        ('SCAB', '(Cười khẩy, ném một cuộn băng gạc cáu bẩn sang giường KID) Tao biết mày là ai. Thằng nhãi Vô Địch từ chối ngã xuống ở hiệp 4. Mày nghĩ xương mày cứng? Ở cái lỗ Golgotha này, xương cứng chỉ làm bọn nó nhai lâu hơn thôi.'),
        ('KID', '(Vừa quấn băng gạc vào tay, vừa nhìn thẳng) Tôi không định chết già ở đây.'),
        ('SCAB', 'Thằng nào mới vào cũng nói thế... cho đến khi chúng nó bị lôi xuống Tầng Hầm. The Shadow Ring.'),
        ('KID', 'Sàn đấu ngầm?'),
        ('SCAB', 'Đúng. Thằng Giám ngục dạo này đang thiếu trò vui cho bọn nhà giàu cá cược. Nó vừa ném ra một mẩu xương mới: Kẻ nào đánh bại được tất cả, bước qua xác con chó săn vô địch của nó... sẽ nhận được một tờ giấy ân xá. Có chữ ký đỏ chót của Thống đốc.'),
        ('KID', '(Khựng lại, nắm chặt tay lại thành nắm đấm) Luật chơi thế nào?'),
        ('SCAB', '(Chồm người tới trước, giọng thì thào đe dọa) Không găng tay. Không trọng tài. Không có hiệp phụ. Chỉ có máu đổi lấy tự do. Mày dám cược cái mạng quèn này không, Vô Địch?'),
        ('KID', '(Đứng dậy, bước về phía khung cửa sắt của buồng giam. Góc máy quay đối diện với cánh cửa đang từ từ mở ra) Bảo bọn chúng xếp hàng đi.'),
    ]

    for speaker, text in dialogue:
        draw_dialogue_box(screen, speaker, text)
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        waiting = False
            clock.tick(FPS)
    return True


def show_tournament_progress(screen, clock, stage_images, current_stage):
    stage_names = ['STAGE 1', 'STAGE 2', 'STAGE 3']
    stage_subtitles = ['ROOKIE BRAWL', 'PRISON ENFORCER', 'THE CHAMPION']
    selected = current_stage

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    selected = max(0, selected - 1)
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    selected = min(len(stage_images) - 1, selected + 1)
                if event.key == pygame.K_RETURN:
                    if selected == current_stage:
                        return current_stage
                if event.key == pygame.K_ESCAPE:
                    return None

        screen.fill((10, 10, 10))
        header_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 50, bold=True)
        label_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 28, bold=True)
        hint_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 22, bold=True)

        header = header_font.render('TOURNAMENT PROGRESSION', True, (220, 220, 220))
        screen.blit(header, (WIDTH // 2 - header.get_width() // 2, 40))

        for idx, bg in enumerate(stage_images):
            thumb = pygame.transform.scale(bg, (330, 200)) if bg else pygame.Surface((330, 200))
            x = 90 + idx * 380
            y = 140
            rect = pygame.Rect(x - 10, y - 10, 350, 220)
            border_color = (255, 220, 100) if idx == selected else (80, 80, 80)
            pygame.draw.rect(screen, border_color, rect, 4, border_radius=10)
            screen.blit(thumb, (x, y))

            status = 'CLEARED' if idx < current_stage else 'NEXT' if idx == current_stage else 'LOCKED'
            status_color = (100, 255, 100) if status == 'CLEARED' else (255, 220, 100) if status == 'NEXT' else (255, 120, 120)

            stage_label = label_font.render(stage_names[idx], True, (255, 255, 255))
            subtitle = label_font.render(stage_subtitles[idx], True, (200, 200, 200))
            status_label = label_font.render(status, True, status_color)

            screen.blit(stage_label, (x + (330 - stage_label.get_width()) // 2, y + 210))
            screen.blit(subtitle, (x + (330 - subtitle.get_width()) // 2, y + 240))
            screen.blit(status_label, (x + (330 - status_label.get_width()) // 2, y + 270))

            if idx > current_stage:
                lock_overlay = pygame.Surface((330, 200), pygame.SRCALPHA)
                lock_overlay.fill((0, 0, 0, 170))
                screen.blit(lock_overlay, (x, y))
                lock_text = label_font.render('LOCKED', True, (255, 120, 120))
                screen.blit(lock_text, (x + (330 - lock_text.get_width()) // 2, y + 80))

        stage_message = 'Chỉ có thể chọn vòng hiện tại. Thắng Stage hiện tại để mở Stage tiếp theo.'
        info_text = hint_font.render(stage_message, True, (190, 190, 190))
        screen.blit(info_text, (WIDTH // 2 - info_text.get_width() // 2, HEIGHT - 90))

        control_text = hint_font.render('LEFT/RIGHT TO BROWSE - ENTER TO FIGHT - ESC TO RETURN', True, (190, 190, 190))
        screen.blit(control_text, (WIDTH // 2 - control_text.get_width() // 2, HEIGHT - 60))

        pygame.display.flip()
        clock.tick(FPS)


def run_story_intro(screen, clock, stage_images, current_stage):
    story_filename = 'STORY-MUSIC.wav'
    story_sound = load_audio(story_filename)
    if story_sound:
        story_sound.play(-1)

    if current_stage == 0:
        video_name = 'Boxer_s_Gritty_Alleyway_Club_Entrance.mp4'
        play_intro_video(screen, clock, video_name)
        if not run_intro_dialogue(screen, clock):
            if story_sound:
                story_sound.stop()
            return None
    if story_sound:
        story_sound.stop()

    return show_tournament_progress(screen, clock, stage_images, current_stage)


def draw_championship_belt(screen, belt_image, glow_progress):
    center = (WIDTH // 2, HEIGHT // 2 - 80)
    glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    glow_strength = 110 + int(40 * math.sin(glow_progress * math.pi * 2))
    glow_color = (255, 220, 80, glow_strength)
    glow_radius = 220 + int(20 * glow_progress)

    glow_rect = pygame.Rect(0, 0, glow_radius * 2, glow_radius)
    glow_rect.center = center
    pygame.draw.ellipse(glow_surface, glow_color, glow_rect)
    glow_surface = pygame.transform.smoothscale(glow_surface, (glow_surface.get_width(), glow_surface.get_height()))
    screen.blit(glow_surface, (0, 0))

    if belt_image:
        belt_rect = belt_image.get_rect(center=center)
        screen.blit(belt_image, belt_rect)
    else:
        fallback_belt = pygame.Surface((780, 220), pygame.SRCALPHA)
        pygame.draw.ellipse(fallback_belt, (212, 175, 55), fallback_belt.get_rect())
        pygame.draw.ellipse(fallback_belt, (120, 80, 20), fallback_belt.get_rect(), 10)
        fb_rect = fallback_belt.get_rect(center=center)
        screen.blit(fallback_belt, fb_rect)


def show_tournament_champion(screen, clock, belt_image=None):
    title_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 82, bold=True)
    body_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma', 'dejavusans'], 32)
    hint_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 24, bold=True)
    text_lines = [
        'Bạn đã đánh bại mọi đối thủ trong Shadow Ring.',
        'Đai vô địch là của bạn. Ánh sáng huy hoàng đang rực sáng vì chiến thắng này.'
    ]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                    return

        screen.fill((12, 10, 18))
        glow_progress = (math.sin(time.time() * 1.6) + 1) / 2
        draw_championship_belt(screen, belt_image, glow_progress)

        title_surf = title_font.render('CHAMPION OF THE SHADOW RING', True, (255, 235, 180))
        title_shadow = title_font.render('CHAMPION OF THE SHADOW RING', True, (20, 20, 20))
        title_x = WIDTH // 2 - title_surf.get_width() // 2
        screen.blit(title_shadow, (title_x + 4, 40 + 4))
        screen.blit(title_surf, (title_x, 40))

        for idx, line in enumerate(text_lines):
            line_surf = body_font.render(line, True, (220, 220, 220))
            screen.blit(line_surf, (WIDTH // 2 - line_surf.get_width() // 2, HEIGHT - 190 + idx * 40))

        hint_surf = hint_font.render('NHẤN [ENTER] HOẶC [ESC] ĐỂ TIẾP TỤC', True, (180, 180, 180))
        screen.blit(hint_surf, (WIDTH // 2 - hint_surf.get_width() // 2, HEIGHT - 70))

        pygame.display.flip()
        clock.tick(FPS)


def draw_victory_screen(screen, selected_index, background):
    """Draw a brutal, high-contrast victory screen"""
    screen.blit(background, (0, 0))
    
    # Phủ màu đen dày đặc hơn
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 230))
    screen.blit(overlay, (0, 0))
    
    title_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 120, bold=True)
    menu_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 50, bold=True)
    hint_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 24, bold=True)

    # Victory Title - Thay vì "VICTORY" nhàm chán, dùng "KNOCKOUT" hoặc "SURVIVED"
    title_text = 'KNOCKOUT!'
    title_shadow = title_font.render(title_text, True, (80, 0, 0))
    title_surf = title_font.render(title_text, True, (255, 50, 50)) # Đỏ gắt
    
    title_x = WIDTH // 2 - title_surf.get_width() // 2
    title_y = 120
    
    screen.blit(title_shadow, (title_x + 8, title_y + 8))
    screen.blit(title_surf, (title_x, title_y))

    # Khung Menu sắc cạnh (Industrial Box)
    panel_width, panel_height = 600, 380
    panel_rect = pygame.Rect(WIDTH // 2 - panel_width // 2, HEIGHT // 2 - panel_height // 2 + 50, panel_width, panel_height)
    
    # Vẽ nền khung (đen trong suốt) và viền (đỏ thẫm, nét vuông vức)
    pygame.draw.rect(screen, (20, 5, 5, 200), panel_rect)
    pygame.draw.rect(screen, (150, 30, 30), panel_rect, 6) # Nét dày, KHÔNG dùng border_radius
    # Thêm các góc nhọn trang trí
    pygame.draw.rect(screen, (255, 50, 50), (panel_rect.left, panel_rect.top, 20, 20))
    pygame.draw.rect(screen, (255, 50, 50), (panel_rect.right - 20, panel_rect.bottom - 20, 20, 20))

    # Menu items
    menu_items = ['NEXT ROUND', 'RETURN TO CELL (MENU)', 'ESCAPE PRISON (EXIT)']
    for index, text in enumerate(menu_items):
        if index == selected_index:
            display_text = f"> {text} <"
            color = (255, 255, 255) # Trắng sáng khi chọn
            bg_rect_color = (150, 30, 30) # Vệt màu đỏ sau chữ
        else:
            display_text = text
            color = (150, 150, 150) # Xám mờ
            bg_rect_color = None

        item_surf = menu_font.render(display_text, True, color)
        item_x = WIDTH // 2 - item_surf.get_width() // 2
        item_y = panel_rect.top + 60 + index * 90
        
        # Vẽ vệt nền (highlight) nếu đang chọn
        if bg_rect_color:
            highlight_rect = pygame.Rect(WIDTH // 2 - 250, item_y - 5, 500, item_surf.get_height() + 10)
            pygame.draw.rect(screen, bg_rect_color, highlight_rect)
            
        screen.blit(item_surf, (item_x, item_y))

    # Hint text
    hint_surf = hint_font.render('YOUR SENTENCE CONTINUES...', True, (100, 100, 100))
    screen.blit(hint_surf, (WIDTH // 2 - hint_surf.get_width() // 2, panel_rect.bottom - 40))
    
    pygame.display.flip()


def victory_loop(screen, clock, stage_background):
    """Handle victory screen selection"""
    selected = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % 3
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % 3
                if event.key == pygame.K_RETURN:
                    if selected == 0:
                        return 'next'
                    elif selected == 1:
                        return 'menu'
                    else:
                        return 'quit'
                if event.key == pygame.K_ESCAPE:
                    return 'menu'

        draw_victory_screen(screen, selected, stage_background)
        clock.tick(FPS)

def draw_defeat_screen(screen, selected_index, background):
    """Màn hình khi người chơi hết máu (Gục ngã)"""
    screen.blit(background, (0, 0))
    
    # Lớp phủ màu đỏ tối (Hiệu ứng máu/bầm dập)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((50, 5, 5, 210))
    screen.blit(overlay, (0, 0))
    
    title_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 130, bold=True)
    menu_font = get_unicode_font(['segoeui', 'arial', 'verdana', 'tahoma'], 50, bold=True)

    title_text = 'KNOCKED OUT!'
    title_shadow = title_font.render(title_text, True, (0, 0, 0))
    title_surf = title_font.render(title_text, True, (200, 30, 30)) 
    
    title_x = WIDTH // 2 - title_surf.get_width() // 2
    title_y = 150
    screen.blit(title_shadow, (title_x + 8, title_y + 8))
    screen.blit(title_surf, (title_x, title_y))

    # Khung Menu
    panel_width, panel_height = 500, 260
    panel_rect = pygame.Rect(WIDTH // 2 - panel_width // 2, HEIGHT // 2 - panel_height // 2 + 80, panel_width, panel_height)
    pygame.draw.rect(screen, (15, 5, 5, 230), panel_rect)
    pygame.draw.rect(screen, (100, 20, 20), panel_rect, 4)

    menu_items = ['GET UP (RESTART)', 'GIVE UP (MENU)']
    for index, text in enumerate(menu_items):
        if index == selected_index:
            display_text = f"> {text} <"
            color = (255, 255, 255) 
            bg_rect_color = (150, 30, 30) 
        else:
            display_text = text
            color = (120, 120, 120)
            bg_rect_color = None

        item_surf = menu_font.render(display_text, True, color)
        item_y = panel_rect.top + 50 + index * 90
        
        if bg_rect_color:
            highlight_rect = pygame.Rect(WIDTH // 2 - 200, item_y - 5, 400, item_surf.get_height() + 10)
            pygame.draw.rect(screen, bg_rect_color, highlight_rect)
            
        screen.blit(item_surf, (WIDTH // 2 - item_surf.get_width() // 2, item_y))

    pygame.display.flip()

def defeat_loop(screen, clock, stage_background):
    """Vòng lặp xử lý lựa chọn khi thua"""
    selected = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % 2 # Có 2 lựa chọn
                if event.key == pygame.K_RETURN:
                    if selected == 0: return 'restart'
                    elif selected == 1: return 'menu'
                if event.key == pygame.K_ESCAPE:
                    return 'menu'

        draw_defeat_screen(screen, selected, stage_background)
        clock.tick(FPS)


def menu_loop(screen, clock, menu_background):
    selected = 0
    
    # Play background music
    if load_music('BACKGROUND MUSIC.mp4'):
        pygame.mixer.music.play(-1)  # Loop infinitely
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mixer.music.stop()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % 2
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % 2
                if event.key == pygame.K_RETURN:
                    pygame.mixer.music.stop()
                    return selected == 0
                if event.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop()
                    return False

        draw_menu(screen, selected, menu_background)
        clock.tick(FPS)


def game_loop(screen, clock, stage_background, enemy_image, enemy_hit_image, enemy_attack_image, enemy_block_image, glove1, glove2, punch_sound, win_sound, current_stage):
    """Main game loop - returns True to restart, False to exit"""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam')
        return False

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    player_health, opponent_health = 100, 100
    last_hit_time = 0.0
    hit_frame_end_time = 0.0
    enemy_is_hit = False



    if current_stage == 0: # STAGE 1: Tân binh
        attack_interval_min, attack_interval_max = 2.0, 4.5
        block_chance = 0.20
        base_attack_power = 8
    elif current_stage == 1: # STAGE 2: Gã khổng lồ ngục tối
        attack_interval_min, attack_interval_max = 1.8, 4.0
        block_chance = 0.65  # Tăng mạnh tỉ lệ đỡ đòn
        base_attack_power = 10
    else: # STAGE 3: Nhà cựu vô địch
        attack_interval_min, attack_interval_max = 0.8, 2.2 # Tốc độ vung đấm cực nhanh
        block_chance = 0.40
        base_attack_power = 16 # Sát thương đấm cực đau

    enemy_last_player_damage = 8
    enemy_attack_power = base_attack_power
    enemy_is_attacking = False
    enemy_attack_executed = False
    enemy_attack_end_time = 0.0
    enemy_blocking = False
    enemy_block_end_time = 0.0

    next_enemy_action_time = time.time() + random.uniform(attack_interval_min, attack_interval_max)

    notifications = []  # List of (text, color, start_time, duration)

    prev_left = None
    prev_right = None
    prev_left_depth = None
    prev_right_depth = None

    # Play stage music
    if load_music('STAGE FIGHT.mp4'):
        pygame.mixer.music.play(-1)  # Loop infinitely

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.mixer.music.stop()
                return 'quit'

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # camera preview + pose overlay for debugging
        debug_frame = frame.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                debug_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
            )

        left_glove = (WIDTH // 2 - 120, HEIGHT - 150)
        right_glove = (WIDTH // 2 + 120, HEIGHT - 150)

        left_punching = False
        right_punching = False
        player_blocking = False
        action_state = 'Normal'

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Left side
            ls = normalize_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], frame_w, frame_h)
            le = normalize_point(lm[mp_pose.PoseLandmark.LEFT_ELBOW], frame_w, frame_h)
            lw = normalize_point(lm[mp_pose.PoseLandmark.LEFT_WRIST], frame_w, frame_h)
            lw_z = lm[mp_pose.PoseLandmark.LEFT_WRIST].z
            # Right side
            rs = normalize_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], frame_w, frame_h)
            re = normalize_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW], frame_w, frame_h)
            rw = normalize_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST], frame_w, frame_h)
            rw_z = lm[mp_pose.PoseLandmark.RIGHT_WRIST].z

            left_screen = map_to_screen(lw, frame_w, frame_h)
            right_screen = map_to_screen(rw, frame_w, frame_h)

            # smoothing with previous positions
            if prev_left:
                left_screen = (int(0.75 * prev_left[0] + 0.25 * left_screen[0]),
                               int(0.75 * prev_left[1] + 0.25 * left_screen[1]))
            if prev_right:
                right_screen = (int(0.75 * prev_right[0] + 0.25 * right_screen[0]),
                                int(0.75 * prev_right[1] + 0.25 * right_screen[1]))

            # velocity in screen space
            if prev_left:
                vel_left = math.hypot(left_screen[0] - prev_left[0], left_screen[1] - prev_left[1])
            else:
                vel_left = 0
            if prev_right:
                vel_right = math.hypot(right_screen[0] - prev_right[0], right_screen[1] - prev_right[1])
            else:
                vel_right = 0

            # forward movement toward camera in depth space
            if prev_left_depth is not None:
                depth_left = prev_left_depth - lw_z
            else:
                depth_left = 0
            if prev_right_depth is not None:
                depth_right = prev_right_depth - rw_z
            else:
                depth_right = 0

            # angle checks for extended arm
            angle_l = angle_between(ls, le, lw)
            angle_r = angle_between(rs, re, rw)

            if (vel_left > PUNCH_SPEED_THRESHOLD or depth_left > PUNCH_DEPTH_THRESHOLD) and angle_l > PUNCH_ANGLE_THRESHOLD:
                left_punching = True
            if (vel_right > PUNCH_SPEED_THRESHOLD or depth_right > PUNCH_DEPTH_THRESHOLD) and angle_r > PUNCH_ANGLE_THRESHOLD:
                right_punching = True

            player_blocking = is_blocking_pose(lm, frame_w, frame_h)
            if player_blocking and not (left_punching or right_punching):
                action_state = 'Blocking'
            elif left_punching and not right_punching:
                action_state = 'Left punch'
            elif right_punching and not left_punching:
                action_state = 'Right punch'
            elif left_punching and right_punching:
                action_state = 'Both punches'

            prev_left = left_screen
            prev_right = right_screen
            prev_left_depth = lw_z
            prev_right_depth = rw_z

            left_glove = left_screen
            right_glove = right_screen

        # enemy AI: attack or block randomly on a timer
        now = time.time()
        if now >= next_enemy_action_time and not enemy_is_attacking and not enemy_blocking:
            if random.random() < block_chance: # Dùng tỉ lệ block theo Stage
                enemy_blocking = True
                enemy_block_end_time = now + ENEMY_BLOCK_DURATION # (Thời gian block bạn đã tăng lên 2.5 trước đó)
                action_state = 'Enemy blocking'
            else:
                enemy_is_attacking = True
                enemy_attack_executed = False
                enemy_attack_end_time = now + ENEMY_ATTACK_DURATION
                # Sát thương tay đôi: Kẻ địch học theo sát thương của bạn hoặc dùng base power
                enemy_attack_power = max(base_attack_power, enemy_last_player_damage)
                action_state = 'Enemy attacking'
            
            # Cập nhật thời gian đấm tiếp theo
            next_enemy_action_time = now + random.uniform(attack_interval_min, attack_interval_max)

        if enemy_is_attacking and now >= enemy_attack_end_time:
            enemy_is_attacking = False
            enemy_attack_executed = False

        if enemy_blocking and now >= enemy_block_end_time:
            enemy_blocking = False

        # rendering
        if stage_background:
            screen.blit(stage_background, (0, 0))
        else:
            screen.fill((20, 20, 20))
            # fallback background (arena floor)
            pygame.draw.rect(screen, (70, 70, 90), (0, HEIGHT // 2, WIDTH, HEIGHT // 2))
            pygame.draw.rect(screen, (180, 180, 180), (0, HEIGHT // 2 - 2, WIDTH, 4))

        # opponent
        if enemy_is_attacking and enemy_attack_image:
            active_enemy_image = enemy_attack_image
        elif enemy_blocking and enemy_block_image:
            active_enemy_image = enemy_block_image
        else:
            active_enemy_image = enemy_hit_image if enemy_is_hit and enemy_hit_image else enemy_image

        if active_enemy_image:
            enemy_rect = active_enemy_image.get_rect(center=(OPP_X, OPP_Y))
            screen.blit(active_enemy_image, enemy_rect)
        else:
            # fallback opponent rendering
            pygame.draw.rect(screen, (200, 50, 50), (OPP_X - OPP_W//2, OPP_Y - OPP_H//2, OPP_W, OPP_H))
            pygame.draw.circle(screen, (220, 180, 120), (OPP_X, OPP_Y - 160), 42)

        # hit effects
        if left_punching or right_punching:
            if time.time() - last_hit_time > HIT_COOLDOWN:
                glove_rects = [pygame.Rect(left_glove[0]-25, left_glove[1]-25, 50, 50),
                               pygame.Rect(right_glove[0]-25, right_glove[1]-25, 50, 50)]
                for glove in glove_rects:
                    if glove.colliderect(HEAD_BOX) or glove.colliderect(BODY_BOX):
                        # Tính sát thương thô (Đầu: 12, Thân: 6)
                        raw_damage = 12 if glove.colliderect(HEAD_BOX) else 6
                        enemy_last_player_damage = raw_damage
                        last_hit_time = time.time()
                        damage_amount = raw_damage

                        if enemy_blocking:
                            if current_stage >= 1: 
                                # Stage 2 & 3: Giảm 60% sát thương (Kẻ địch chịu 40%)
                                damage_amount = int(raw_damage * 0.4)
                                notifications.append((f'GUARD BROKEN! -{damage_amount} HP', (255, 150, 0), time.time(), 2.0))
                            else:
                                # Stage 1: Đỡ được 100% sát thương
                                damage_amount = 0
                                notifications.append(('ENEMY BLOCKED!', (255, 255, 0), time.time(), 2.0))
                            action_state = 'Enemy blocked'
                        else:
                            notifications.append((f'ENEMY HIT! -{damage_amount} HP', (255, 0, 0), time.time(), 2.0))
                        
                        # Chỉ gây máu và tạo hiệu ứng chớp nếu sát thương > 0
                        if damage_amount > 0:
                            opponent_health -= damage_amount
                            enemy_is_hit = True
                            hit_frame_end_time = time.time() + 0.25
                            if punch_sound:
                                punch_sound.play()
                        break

        # enemy attack hits player
        if enemy_is_attacking and not enemy_attack_executed:
            enemy_attack_executed = True
            if player_blocking:
                blocked_damage = int(enemy_attack_power * 0.4)
                player_health -= blocked_damage
                if blocked_damage > 0:
                    notifications.append((f'PLAYER BLOCKED! -{blocked_damage} HP', (0, 255, 0), time.time(), 2.0))
                    action_state = 'Enemy hit block'
                else:
                    notifications.append(('PLAYER PERFECT BLOCK!', (0, 255, 0), time.time(), 2.0))
                    action_state = 'Player blocked'
            else:
                player_health -= enemy_attack_power
                notifications.append((f'PLAYER HIT! -{enemy_attack_power} HP', (255, 0, 0), time.time(), 2.0))
                action_state = 'Enemy landed punch'

        # draw player gloves
        draw_glove_at_position(screen, glove1, left_glove)
        draw_glove_at_position(screen, glove2, right_glove)

        # health bars
        pygame.draw.rect(screen, (80, 80, 80), (50, 30, 400, 25))
        pygame.draw.rect(screen, (20, 220, 20), (50, 30, max(0, 4 * player_health), 25))
        pygame.draw.rect(screen, (80, 80, 80), (WIDTH-450, 30, 400, 25))
        pygame.draw.rect(screen, (220, 20, 20), (WIDTH-450, 30, max(0, 4 * opponent_health), 25))

        font = get_unicode_font(['arial', 'verdana', 'tahoma'], 32)
        screen.blit(font.render('Player', True, (255,255,255)), (50, 60))
        screen.blit(font.render('Opponent', True, (255,255,255)), (WIDTH-170, 60))
        screen.blit(font.render(f'State: {action_state}', True, (255,255,255)), (50, 100))
        enemy_status = 'ATTACK' if enemy_is_attacking else 'BLOCK' if enemy_blocking else 'IDLE'
        screen.blit(font.render(f'Enemy: {enemy_status}', True, (255, 255, 255)), (50, 140))

        # time bar / info
        screen.blit(font.render('Press ESC to quit', True, (220,220,220)), (WIDTH//2-100, 20))

        # draw camera preview on the side for debugging skeleton/hand detection
        cam_w, cam_h = 320, 180
        debug_small = cv2.resize(debug_frame, (cam_w, cam_h))
        debug_small = cv2.cvtColor(debug_small, cv2.COLOR_BGR2RGB)
        debug_surf = pygame.image.frombuffer(debug_small.tobytes(), (cam_w, cam_h), 'RGB')
        screen.blit(debug_surf, (WIDTH - cam_w - 20, 80))
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH - cam_w - 22, 78, cam_w + 4, cam_h + 4), 2)
        screen.blit(font.render('Webcam (debug)', True, (255, 255, 255)), (WIDTH - cam_w - 18, 55))

        # render notifications
        now = time.time()
        y_offset = 180
        for i, (text, color, start_time, duration) in enumerate(notifications[:]):
            elapsed = now - start_time
            if elapsed > duration:
                notifications.remove((text, color, start_time, duration))
                continue
            
            # Calculate fade alpha with smoother curve
            fade_progress = min(elapsed / duration, 1.0)
            alpha = max(0, int(255 * (1 - fade_progress ** 2)))  # Quadratic fade for smoother effect
            
            # Scale effect (pop): scale from 0.8 to 1.0 over first 0.2 seconds
            pop_duration = 0.2
            if elapsed < pop_duration:
                scale = 0.8 + (0.2 * (elapsed / pop_duration))
            else:
                scale = 1.0
            
            # Float up effect: move upward over time
            float_offset = elapsed * 30  # pixels per second upward
            
            notif_font = get_unicode_font(['arial', 'verdana', 'tahoma'], 36, bold=True)
            
            # Render main text
            notif_surf = notif_font.render(text, True, color)
            notif_surf.set_alpha(alpha)
            
            # Apply scale
            if scale != 1.0:
                new_width = int(notif_surf.get_width() * scale)
                new_height = int(notif_surf.get_height() * scale)
                notif_surf = pygame.transform.smoothscale(notif_surf, (new_width, new_height))
            
            # Calculate position with float
            x_pos = WIDTH // 2 - notif_surf.get_width() // 2
            y_pos = y_offset - float_offset
            
            # Render shadow (offset by 3 pixels, darker)
            shadow_surf = notif_font.render(text, True, (0, 0, 0))
            shadow_surf.set_alpha(int(alpha * 0.5))
            if scale != 1.0:
                shadow_surf = pygame.transform.smoothscale(shadow_surf, (new_width, new_height))
            screen.blit(shadow_surf, (x_pos + 3, y_pos + 3))
            
            # Render outline (4 directions)
            outline_color = (0, 0, 0)
            outline_surf = notif_font.render(text, True, outline_color)
            outline_surf.set_alpha(alpha)
            if scale != 1.0:
                outline_surf = pygame.transform.smoothscale(outline_surf, (new_width, new_height))
            
            # Draw outline in 4 directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                screen.blit(outline_surf, (x_pos + dx, y_pos + dy))
            
            # Render main text on top
            screen.blit(notif_surf, (x_pos, y_pos))
            
            y_offset += 50

        if enemy_is_hit and time.time() > hit_frame_end_time:
            enemy_is_hit = False

        pygame.display.flip()

        if opponent_health <= 0:
            pygame.mixer.music.stop()
            if win_sound:
                win_sound.play()
            print('Opponent defeated!')
            time.sleep(1)  # Brief pause before showing victory screen

            victory_choice = victory_loop(screen, clock, stage_background)
            cap.release()
            pygame.mixer.music.stop()

            if victory_choice == 'next':
                return 'next'
            elif victory_choice == 'menu':
                return 'menu'
            else:
                return 'quit'
            pass

        if player_health <= 0:
            pygame.mixer.music.stop()
            print('Player defeated!')
            time.sleep(1) # Dừng 1 nhịp để thấy máu đã cạn

            # Mở màn hình Thất bại
            defeat_choice = defeat_loop(screen, clock, stage_background)
            cap.release()
            pygame.mixer.music.stop()

            # Trả về kết quả cho hàm main() xử lý
            if defeat_choice == 'restart':
                return 'restart'
            elif defeat_choice == 'menu':
                return 'menu'
            else:
                return 'quit'

        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False

        clock.tick(FPS)

    cap.release()
    pygame.mixer.music.stop()
    return 'menu'

def load_enemy_assets_for_stage(stage_index):
    """Load và scale hình ảnh kẻ địch tương ứng với từng Stage"""
    if stage_index == 0:  # Stage 1
        img_stand = 'enemystage1.png'
        img_hit = 'BEINGHITEFFECTSTAGE1.png'
        img_punch = 'PUNCHEFFECTSTAGE1.png'
        img_block = 'BLOCKINGEFFECTSTAGE1.png'
    elif stage_index == 1:  # Stage 2 (Juggernaut)
        img_stand = 'STANDINGENEMY2.png'
        img_hit = 'BEINGHITENEMY2.png'
        img_punch = 'PUNCHENEMY2.png'
        img_block = 'BLOCKINGENEMY2.png'
    else:  # Stage 3 (Champion)
        img_stand = 'STANDINGENEMY3.png'
        img_hit = 'BEINGHITENEMY3.png'
        img_punch = 'PUNCHENEMY3.png'
        img_block = 'BLOCKINGENEMY3.png'

    # Load ảnh cơ bản
    enemy_image = load_image(img_stand, (OPP_W, OPP_H))
    enemy_hit_image = load_image(img_hit, (OPP_W, OPP_H))
    enemy_attack_image = load_image(img_punch, (OPP_W, OPP_H))
    enemy_block_image = load_image(img_block, (OPP_W, OPP_H))

    # Scale ảnh như logic cũ để khớp kích thước khung hình
    if enemy_image and enemy_hit_image:
        enemy_hit_image = pygame.transform.smoothscale(
            enemy_hit_image,
            (int(OPP_W * 1.2), int(OPP_H * 1.2))
        ).convert_alpha()

    if enemy_attack_image:
        enemy_attack_image = pygame.transform.smoothscale(
            enemy_attack_image,
            (int(OPP_W * 1.35), int(OPP_H * 1.35))
        ).convert_alpha()

    if enemy_block_image:
        enemy_block_image = pygame.transform.smoothscale(
            enemy_block_image,
            (int(OPP_W * 1.35), int(OPP_H * 1.35))
        ).convert_alpha()

    return enemy_image, enemy_hit_image, enemy_attack_image, enemy_block_image

def main():
    global WIDTH, HEIGHT
    global OPP_X, OPP_Y, OPP_W, OPP_H, HEAD_BOX, BODY_BOX

    pygame.init()
    pygame.mixer.init()

    # ✅ FULLSCREEN CHUẨN
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    WIDTH, HEIGHT = screen.get_size()

    pygame.display.set_caption('Shadow Boxer AI (FPP Boxing)')
    clock = pygame.time.Clock()

    # ✅ SET LẠI HITBOX SAU KHI CÓ WIDTH HEIGHT
    OPP_W, OPP_H = 330, 570
    OPP_X, OPP_Y = WIDTH // 2, HEIGHT // 2

    HEAD_BOX = pygame.Rect(OPP_X - 60, OPP_Y - 240, 120, 120)
    BODY_BOX = pygame.Rect(OPP_X - 105, OPP_Y - 120, 210, 270)

    # Background menu
    img_path = os.path.join(os.path.dirname(__file__), 'IMGS', 'anh3.png')
    if os.path.exists(img_path):
        menu_background = pygame.image.load(img_path).convert()
        menu_background = pygame.transform.scale(menu_background, (WIDTH, HEIGHT))
    else:
        menu_background = pygame.Surface((WIDTH, HEIGHT))
        menu_background.fill((12, 12, 20))

    # Load stage images
    stage_images = [
        load_image('Stage1.png', (WIDTH, HEIGHT)),
        load_image('Stage2.png', (WIDTH, HEIGHT)),
        load_image('Stage3.png', (WIDTH, HEIGHT)),
    ]

    # enemy_image = load_image('enemystage1.png', (OPP_W, OPP_H))
    # enemy_hit_image = load_image('BEINGHITEFFECTSTAGE1.png', (OPP_W, OPP_H))
    # enemy_attack_image = load_image('PUNCHEFFECTSTAGE1.png', (OPP_W, OPP_H))
    # enemy_block_image = load_image('BLOCKINGEFFECTSTAGE1.png', (OPP_W, OPP_H))

    championship_belt_image = load_image('championship_belt.png')

    # Resize belt
    if championship_belt_image:
        belt_target_width = min(WIDTH - 220, championship_belt_image.get_width())
        belt_target_height = int(championship_belt_image.get_height() * (belt_target_width / championship_belt_image.get_width()))
        championship_belt_image = pygame.transform.smoothscale(
            championship_belt_image, (belt_target_width, belt_target_height)
        )

    # # Scale enemy states
    # if enemy_image and enemy_hit_image:
    #     enemy_hit_image = pygame.transform.smoothscale(
    #         enemy_hit_image,
    #         (int(OPP_W * 1.2), int(OPP_H * 1.2))
    #     ).convert_alpha()

    # if enemy_attack_image:
    #     enemy_attack_image = pygame.transform.smoothscale(
    #         enemy_attack_image,
    #         (int(OPP_W * 1.35), int(OPP_H * 1.35))
    #     ).convert_alpha()

    # if enemy_block_image:
    #     enemy_block_image = pygame.transform.smoothscale(
    #         enemy_block_image,
    #         (int(OPP_W * 1.35), int(OPP_H * 1.35))
    #     ).convert_alpha()

    # Gloves
    glove1 = load_glove_image('GLOVE2.PNG', size=90)
    glove2 = load_glove_image('GLOVE1.png', size=90)

    # Audio
    punch_sound = load_audio('PUNCH EFECT.wav')
    win_sound = load_audio('WIN EFECT.mp4')

    # ================= MAIN LOOP =================
    while True:
        if not menu_loop(screen, clock, menu_background):
            break

        current_stage = 0
        while current_stage < len(stage_images):
            selected_stage = run_story_intro(screen, clock, stage_images, current_stage)
            if selected_stage is None:
                break

            stage_background = stage_images[selected_stage]

        
            e_img, e_hit, e_atk, e_blk = load_enemy_assets_for_stage(current_stage)

            result = game_loop(
                screen, clock,
                stage_background,
                e_img, e_hit, e_atk, e_blk,
                glove1, glove2,
                punch_sound, win_sound,
                current_stage  # <---- CHÍNH LÀ BIẾN NÀY
            )

            if result == 'next':
                current_stage += 1
                if current_stage >= len(stage_images):
                    show_tournament_champion(screen, clock, championship_belt_image)
                    break
            elif result == 'menu':
                break
            elif result == 'quit':
                pygame.quit()
                return
            else:
                break

    pygame.quit()

if __name__ == '__main__':
    main()
