import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.transforms.functional import rgb_to_grayscale
from itertools import cycle
from numpy.random import randint
from pygame import Rect, init, time, display, event
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate
import numpy as np
import os
import glob
import csv

# ==========================================
# 1. CLASS FLAPPY BIRD (Versi SUPER CEPAT)
# ==========================================
class FlappyBirdFast(object):
    init()
    screen_width = 288
    screen_height = 512
    screen = display.set_mode((screen_width, screen_height))
    display.set_caption('Auto-Evaluasi Massal')
    
    img_folder = 'I:\\Flappy Bird AI\\'
    base_image = load(f'{img_folder}assets/sprites/base.png').convert_alpha()
    background_image = load(f'{img_folder}assets/sprites/background-black.png').convert()

    pipe_images = [rotate(load(f'{img_folder}assets/sprites/pipe-green.png').convert_alpha(), 180),
                   load(f'{img_folder}assets/sprites/pipe-green.png').convert_alpha()]
    bird_images = [load(f'{img_folder}assets/sprites/redbird-upflap.png').convert_alpha(),
                   load(f'{img_folder}assets/sprites/redbird-midflap.png').convert_alpha(),
                   load(f'{img_folder}assets/sprites/redbird-downflap.png').convert_alpha()]

    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]

    pipe_gap_size = 100
    pipe_velocity_x = -4
    min_velocity_y = -8
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -9
    bird_index_generator = cycle([0, 1, 2, 1])

    def __init__(self):
        self.iter = self.bird_index = self.score = 0
        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()
        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)
        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()
        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.pipes = pipes
        self.current_velocity_y = 0
        self.is_flapped = False

    def generate_pipe(self):
        x = self.screen_width + 10
        gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        return {"x_upper": x, "y_upper": gap_y - self.pipe_height, "x_lower": x, "y_lower": gap_y + self.pipe_gap_size}

    def is_collided(self):
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        pipe_boxes = []
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe["x_upper"], pipe["y_upper"], self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe["x_lower"], pipe["y_lower"], self.pipe_width, self.pipe_height))
            if bird_bbox.collidelist(pipe_boxes) == -1:
                return False
            for i in range(2):
                cropped_bbox = bird_bbox.clip(pipe_boxes[i])
                min_x1 = cropped_bbox.x - bird_bbox.x
                min_y1 = cropped_bbox.y - bird_bbox.y
                min_x2 = cropped_bbox.x - pipe_boxes[i].x
                min_y2 = cropped_bbox.y - pipe_boxes[i].y
                if np.any(self.bird_hitmask[self.bird_index][min_x1:min_x1 + cropped_bbox.width,
                       min_y1:min_y1 + cropped_bbox.height] * self.pipe_hitmask[i][min_x2:min_x2 + cropped_bbox.width,
                                                                      min_y2:min_y2 + cropped_bbox.height]):
                    return True
        return False

    def next_frame(self, action):
        event.pump()
        reward = 0.1
        terminal = False
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True

        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1
                break

        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.base_x = -((-self.base_x + 100) % self.base_shift)

        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        if self.is_flapped:
            self.is_flapped = False
        self.bird_y += min(self.current_velocity_y, self.bird_y - self.current_velocity_y - self.bird_height)
        if self.bird_y < 0:
            self.bird_y = 0

        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x
        
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]
            
        if self.is_collided():
            terminal = True
            reward = -1

        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        
        image = array3d(display.get_surface()) 
        return image, reward, terminal

# ==========================================
# 2. CLASS DEEP Q-NETWORK
# ==========================================
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

def preprocess(image, game_state):
    img = rgb_to_grayscale(transforms.Resize((84, 84))(
        torch.from_numpy(image[:game_state.screen_width, :int(game_state.base_y)]).permute(2, 1, 0)
    ))
    threshold = 1
    binary_img = (img < threshold).float()
    return binary_img

# ==========================================
# 3. SCRIPT GENERATOR LAPORAN CSV
# ==========================================
def generate_report():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Memulai Evaluasi Massal menggunakan device: {device}\n')
    
    # Mengambil semua checkpoint dan mengurutkannya secara logis (berdasarkan angka)
    ckpt_files = glob.glob('trained_models/checkpoint_*.pt')
    ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not ckpt_files:
        print("Tidak ada file checkpoint ditemukan!")
        return

    N_PERCOBAAN = 5       # Tiap checkpoint dimainkan 5 kali untuk dicari rata-ratanya
    SKOR_MAKSIMAL = 999   # Batas aman agar loop tidak stuck jika AI sudah sangat jago
    FILE_OUTPUT = 'laporan_performa_flappybird.csv'
    
    model = DeepQNetwork().to(device)
    
    # Membuat dan membuka file CSV
    with open(FILE_OUTPUT, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Menulis Header Tabel
        writer.writerow(['Iterasi Checkpoint', 'Rata-rata Skor', 'Skor Tertinggi', 'Skor Terendah'])
        
        # Looping untuk setiap file checkpoint (Dari 20k sampai 1Jt)
        for ckpt_path in ckpt_files:
            iterasi = int(ckpt_path.split('_')[-1].split('.')[0])
            print(f"Mengevaluasi Iterasi {iterasi:,}...", end=" ")
            
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            skor_kumpulan = []
            
            # Bermain sebanyak N kali
            for _ in range(N_PERCOBAAN):
                game_state = FlappyBirdFast()
                image, reward, terminal = game_state.next_frame(0)
                image = preprocess(image, game_state)
                state = torch.cat([image for _ in range(4)], dim=0)[None, ...].to(device)
                
                with torch.no_grad():
                    while not terminal:
                        prediction = model(state)[0]
                        action = torch.argmax(prediction).item()
                        
                        skor_akhir = game_state.score
                        
                        next_image, reward, terminal = game_state.next_frame(action)
                        next_image = preprocess(next_image, game_state).to(device)
                        state = torch.cat((state[0, 1:, :, :], next_image), dim=0)[None, ...].to(device)
                        
                        if skor_akhir >= SKOR_MAKSIMAL:
                            terminal = True
                
                skor_kumpulan.append(skor_akhir)
            
            # Kalkulasi Statistik
            rata_rata = sum(skor_kumpulan) / N_PERCOBAAN
            skor_tertinggi = max(skor_kumpulan)
            skor_terendah = min(skor_kumpulan)
            
            print(f"Rata-rata: {rata_rata} | Max: {skor_tertinggi} | Min: {skor_terendah}")
            
            # Menulis hasil 1 baris ke Excel/CSV
            writer.writerow([iterasi, rata_rata, skor_tertinggi, skor_terendah])

    print("\n" + "="*55)
    print(f"✅ EVALUASI MASSAL SELESAI!")
    print(f"📂 File tabel berhasil disimpan di: {os.path.abspath(FILE_OUTPUT)}")
    print("="*55)

generate_report()