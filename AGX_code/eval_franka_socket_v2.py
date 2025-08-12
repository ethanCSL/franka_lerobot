#server.py
import socket
import os
import struct
import time
import torch
import numpy as np
import cv2
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


class ACTInferenceServer:
    def __init__(self, ckpt_path, host='0.0.0.0', port=5001, device='cuda'):
        # 1. 設定裝置
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，自動切換為 CPU")
            self.device = 'cpu'

        # 2. 絕對路徑轉換
        print("Using:",self.device)
        ckpt_path = os.path.abspath(ckpt_path)
        print(f"📂 使用的 ckpt_path: {ckpt_path}")

        # 3. 使用官方推薦方式載入 config.json 與權重
        #    需確保 pretrained_model 資料夾底下有 config.json 和 model.safetensors
        try:
            # self.policy = ACTPolicy.from_pretrained(
            #     ckpt_path,
            #     local_files_only=True
            # )
            self.policy = DiffusionPolicy.from_pretrained(
                ckpt_path,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"載入 ACTPolicy 失敗，請確認 ckpt 路徑及檔案：{e}")

        # 4. 移到裝置並設為 eval
        self.policy.to(self.device)
        self.policy.eval()

        # 5. 建立 socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print("🚀 ACT 推論伺服器已啟動，等待 client 連線...")
        self.conn, self.addr = self.server_socket.accept()
        print("✅ client 已連線:", self.addr)

    def recv_data(self):
        raw_len = self.conn.recv(4)
        if not raw_len:
            return None, None
        data_len = struct.unpack('>I', raw_len)[0]
        data_type = self.conn.recv(10).decode().strip()
        data = b''
        while len(data) < data_len:
            packet = self.conn.recv(data_len - len(data))
            if not packet:
                break
            data += packet
        return data_type, data

    def send_data(self, data_type, data_bytes):
        data_type = data_type.ljust(10).encode()
        data_len = struct.pack('>I', len(data_bytes))
        self.conn.sendall(data_len + data_type + data_bytes)

    def run(self):
        last_time = time.time() 
        while True:
            try:
                # 接收狀態 list
                type1, data1 = self.recv_data()
                if type1 != 'list':
                    continue
                state = json.loads(data1.decode())
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                # 接收影像
                type2, data2 = self.recv_data()
                if type2 != 'img1':
                    continue
                img1 = cv2.imdecode(np.frombuffer(data2, np.uint8), cv2.IMREAD_COLOR)

                type3, data3 = self.recv_data()
                if type3 != 'img2':
                    continue
                img2 = cv2.imdecode(np.frombuffer(data3, np.uint8), cv2.IMREAD_COLOR)

                # 建立觀測 dict
                obs = {
                    'observation.state': state_tensor,
                    'observation.images.image': torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0,
                    'observation.images.image_additional_view': torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
                }

                # 推論
                with torch.no_grad():
                    action = self.policy.select_action(obs)
                action_np = action.squeeze(0).cpu().numpy().tolist()
                # 計算頻率
                current_time = time.time()              
                interval = current_time - last_time    
                last_time = current_time                
                freq = 1.0 / interval if interval > 0 else 0

                print(f"推論完成: {np.round(action_np,15)} | 頻率: {freq:.2f} Hz")

                # 回傳結果
                self.send_data('list', json.dumps(action_np).encode())

            except Exception as e:
                print("❌ 發生錯誤:", e)
                break

        self.conn.close()
        self.server_socket.close()


if __name__ == '__main__':
    import json
    ckpt_path = '/home/csl/lerobot/outputs/train/0725_red_gripper_change_action_diffusion/checkpoints/last/pretrained_model'
    #ckpt_path = '/home/csl/lerobot/outputs/train/0721/140000/pretrained_model'
    server = ACTInferenceServer(ckpt_path, port=5001, device='cuda')
    server.run()
