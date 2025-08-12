# original  python franka_socket_test.py  --ckpt-path /home/csl/lerobot/outputs/train/0725_red_gripper_change_action/checkpoints/last/pretrained_model/ --eval-freq 10
import socket
import os
import struct
import time
import torch
import numpy as np
from datetime import datetime
import cv2
import argparse
import json
from lerobot.common.policies.act.modeling_act import ACTPolicy


class ACTInferenceServer:
    def __init__(self, ckpt_path, host='0.0.0.0', port=5001, device='cuda', eval_freq=None):
        self.eval_freq = eval_freq  # In Hz
        self.eval_interval = 1.0 / eval_freq if eval_freq and eval_freq > 0 else 0  # seconds

        # 1. Device setup
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，自動切換為 CPU")
            self.device = 'cpu'

        print("Using:", self.device)
        ckpt_path = os.path.abspath(ckpt_path)
        print(f"📂 使用的 ckpt_path: {ckpt_path}")

        try:
            self.policy = ACTPolicy.from_pretrained(
                ckpt_path,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"載入 ACTPolicy 失敗，請確認 ckpt 路徑及檔案：{e}")

        self.policy.to(self.device)
        self.policy.eval()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print("🚀 ACT 推論伺服器已啟動，等待 client 連線...")
        self.conn, self.addr = self.server_socket.accept()
        print("✅ client 已連線:", self.addr)

    def recv_data(self):
        start_time = time.time()
        raw_len = self.conn.recv(4)
        if not raw_len:
            return None, None, 0, 0
        data_len = struct.unpack('>I', raw_len)[0]
        data_type = self.conn.recv(10).decode().strip()
        data = b''
        while len(data) < data_len:
            packet = self.conn.recv(data_len - len(data))
            if not packet:
                break
            data += packet
        end_time = time.time()
        return data_type, data, data_len, end_time - start_time

    def send_data(self, data_type, data_bytes):
        data_type = data_type.ljust(10).encode()
        data_len = struct.pack('>I', len(data_bytes))
        self.conn.sendall(data_len + data_type + data_bytes)

    def run(self):
        last_time = time.time()
        while True:
            try:
                loop_start_time = time.time()
                processing_start_time = loop_start_time

                type1, data1, size1, time1 = self.recv_data()
                if type1 != 'list':
                    continue
                state = json.loads(data1.decode())
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                type2, data2, size2, time2 = self.recv_data()
                if type2 != 'img1':
                    continue
                img1 = cv2.imdecode(np.frombuffer(data2, np.uint8), cv2.IMREAD_COLOR)

                type3, data3, size3, time3 = self.recv_data()
                if type3 != 'img2':
                    continue
                img2 = cv2.imdecode(np.frombuffer(data3, np.uint8), cv2.IMREAD_COLOR)

                obs = {
                    'observation.state': state_tensor,
                    'observation.images.image': torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0,
                    'observation.images.image_additional_view': torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                }

                inference_start_time = time.time()
                with torch.no_grad():
                    action = self.policy.select_action(obs)
                inference_end_time = time.time()

                action_np = action.squeeze(0).cpu().numpy().tolist()

                current_time = time.time()
                total_interval = current_time - last_time
                last_time = current_time
                total_freq = 1.0 / total_interval if total_interval > 0 else 0
                inference_freq = 1.0 / (inference_end_time - inference_start_time) if inference_end_time > inference_start_time else 0
                processing_interval = inference_start_time - processing_start_time
                total_bytes = size1 + size2 + size3
                total_transfer_time = time1 + time2 + time3
                transfer_speed = total_bytes / total_transfer_time if total_transfer_time > 0 else 0

                print(f"推論完成: {np.round(action_np[7],6)} | "
                      f"總頻率: {total_freq:.2f} Hz | "
                      f"純推論頻率: {inference_freq:.2f} Hz | "
                      f"資料處理耗時: {processing_interval:.4f} s | "
                      f"傳輸速率: {transfer_speed / 1024 / 1024:.2f} MB/s")

                self.send_data('list', json.dumps(action_np).encode())

                # --- 控制推論頻率 ---
                if self.eval_interval > 0:
                    elapsed = time.time() - loop_start_time
                    sleep_time = self.eval_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as e:
                print("❌ 發生錯誤:", e)
                break

        self.conn.close()
        self.server_socket.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to pretrained_model folder')
    parser.add_argument('--port', type=int, default=5001, help='Port number')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference')
    parser.add_argument('--eval-freq', type=float, default=0, help='Evaluation frequency in Hz (0 for max speed)')
    args = parser.parse_args()

    server = ACTInferenceServer(
        ckpt_path=args.ckpt_path,
        port=args.port,
        device=args.device,
        eval_freq=args.eval_freq
    )
    server.run()


# import socket
# import os
# import struct
# import time
# import torch
# import numpy as np
# from datetime import datetime
# import cv2
# from lerobot.common.policies.act.modeling_act import ACTPolicy


# class ACTInferenceServer:
#     def __init__(self, ckpt_path, host='0.0.0.0', port=5001, device='cuda'):
#         # 1. 設定裝置
#         self.device = device
#         if device == 'cuda' and not torch.cuda.is_available():
#             print("⚠️ CUDA 不可用，自動切換為 CPU")
#             self.device = 'cpu'

#         # 2. 絕對路徑轉換
#         print("Using:",self.device)
#         ckpt_path = os.path.abspath(ckpt_path)
#         print(f"📂 使用的 ckpt_path: {ckpt_path}")

#         # 3. 使用官方推薦方式載入 config.json 與權重
#         #    需確保 pretrained_model 資料夾底下有 config.json 和 model.safetensors
#         try:
#             self.policy = ACTPolicy.from_pretrained(
#                 ckpt_path,
#                 local_files_only=True
#             )
#         except Exception as e:
#             raise RuntimeError(f"載入 ACTPolicy 失敗，請確認 ckpt 路徑及檔案：{e}")

#         # 4. 移到裝置並設為 eval
#         self.policy.to(self.device)
#         self.policy.eval()


#         # 5. 建立 socket server
#         self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.server_socket.bind((host, port))
#         self.server_socket.listen(1)
#         print("🚀 ACT 推論伺服器已啟動，等待 client 連線...")
#         self.conn, self.addr = self.server_socket.accept()
#         print("✅ client 已連線:", self.addr)

#     def recv_data(self):
#         start_time = time.time()
#         raw_len = self.conn.recv(4)
#         if not raw_len:
#             return None, None, 0, 0
#         data_len = struct.unpack('>I', raw_len)[0]
#         data_type = self.conn.recv(10).decode().strip()
#         data = b''
#         while len(data) < data_len:
#             packet = self.conn.recv(data_len - len(data))
#             if not packet:
#                 break
#             data += packet
#         end_time = time.time()
#         return data_type, data, data_len, end_time - start_time



#     def send_data(self, data_type, data_bytes):
#         data_type = data_type.ljust(10).encode()
#         data_len = struct.pack('>I', len(data_bytes))
#         self.conn.sendall(data_len + data_type + data_bytes)

#     def run(self):
#         last_time = time.time()
#         while True:
#             try:
#                 # --- 步驟 1: 測量接收與前處理時間 ---
#                 processing_start_time = time.time()

#                 # 接收狀態 list
#                 type1, data1, size1, time1 = self.recv_data()
#                 if type1 != 'list':
#                     continue
#                 state = json.loads(data1.decode())
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

#                 # 接收影像 img1
#                 type2, data2, size2, time2 = self.recv_data()
#                 if type2 != 'img1':
#                     continue
#                 img1 = cv2.imdecode(np.frombuffer(data2, np.uint8), cv2.IMREAD_COLOR)

#                 # 接收影像 img2
#                 type3, data3, size3, time3 = self.recv_data()
#                 if type3 != 'img2':
#                     continue
#                 img2 = cv2.imdecode(np.frombuffer(data3, np.uint8), cv2.IMREAD_COLOR)

#                 # 建立觀測 dict
#                 obs = {
#                     'observation.state': state_tensor,
#                     'observation.images.image': torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0,
#                     'observation.images.image_additional_view': torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
#                 }

#                 # --- 步驟 2: 精確測量模型推論時間 ---
#                 inference_start_time = time.time()
#                 with torch.no_grad():
#                     action = self.policy.select_action(obs)
#                 inference_end_time = time.time()

#                 action_np = action.squeeze(0).cpu().numpy().tolist()

#                 # --- 步驟 3: 計算與輸出各種時間與效能指標 ---
#                 current_time = time.time()
#                 total_interval = current_time - last_time
#                 last_time = current_time
#                 total_freq = 1.0 / total_interval if total_interval > 0 else 0

#                 inference_interval = inference_end_time - inference_start_time
#                 inference_freq = 1.0 / inference_interval if inference_interval > 0 else 0

#                 processing_interval = inference_start_time - processing_start_time

#                 # 傳輸速度 (Byte/s -> MB/s)
#                 total_bytes = size1 + size2 + size3
#                 total_transfer_time = time1 + time2 + time3
#                 transfer_speed = total_bytes / total_transfer_time if total_transfer_time > 0 else 0  # Byte/s

#                 print(f"推論完成: {np.round(action_np[7],6)} | "
#                     f"總頻率: {total_freq:.2f} Hz | "
#                     f"純推論頻率: {inference_freq:.2f} Hz | "
#                     f"資料處理耗時: {processing_interval:.4f} s | "
#                     f"傳輸速率: {transfer_speed/1024/1024:.2f} MB/s")

#                 # --- 步驟 4: 傳送推論結果回 client ---
#                 self.send_data('list', json.dumps(action_np).encode())

#             except Exception as e:
#                 print("❌ 發生錯誤:", e)
#                 break

#         self.conn.close()
#         self.server_socket.close()


# if __name__ == '__main__':
#     import json
#     ckpt_path = '/home/csl/lerobot/outputs/train/0725_red_gripper_change_action/checkpoints/last/pretrained_model'
#     server = ACTInferenceServer(ckpt_path, port=5001, device='cuda')
#     server.run()



# import socket
# import os
# import struct
# import time
# import torch
# import numpy as np
# import cv2
# import json # 確保 json 被 import
# from lerobot.common.policies.act.modeling_act import ACTPolicy


# class ACTInferenceServer:
#     def __init__(self, ckpt_path, host='0.0.0.0', port=5001, device='cuda'):
#         # ... (這部分與你原本的程式碼相同，保持不變)
#         # 1. 設定裝置
#         self.device = device
#         if device == 'cuda' and not torch.cuda.is_available():
#             print("⚠️ CUDA 不可用，自動切換為 CPU")
#             self.device = 'cpu'

#         # 2. 絕對路徑轉換
#         print("Using:",self.device)
#         ckpt_path = os.path.abspath(ckpt_path)
#         print(f"📂 使用的 ckpt_path: {ckpt_path}")

#         # 3. 載入模型
#         try:
#             self.policy = ACTPolicy.from_pretrained(
#                 ckpt_path,
#                 local_files_only=True
#             )
#         except Exception as e:
#             raise RuntimeError(f"載入 ACTPolicy 失敗，請確認 ckpt 路徑及檔案：{e}")

#         # 4. 移到裝置並設為 eval
#         self.policy.to(self.device)
#         self.policy.eval()

#         # 5. 建立 socket server
#         self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
#         # [優化建議 1] 設定 TCP_NODELAY 以降低延遲
#         self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
#         self.server_socket.bind((host, port))
#         self.server_socket.listen(1)
#         print("🚀 ACT 推論伺服器已啟動，等待 client 連線...")
#         self.conn, self.addr = self.server_socket.accept()
        
#         # 當 client 連線後也設定 TCP_NODELAY
#         self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
#         print("✅ client 已連線:", self.addr)

#     def _recv_all(self, n):
#         """一個輔助函式，確保接收到指定長度的數據"""
#         data = b''
#         while len(data) < n:
#             packet = self.conn.recv(n - len(data))
#             if not packet:
#                 return None
#             data += packet
#         return data

#     def recv_observation_data(self):
#         """
#         一次性接收所有觀測資料 (state, img1, img2)
#         協定: [4 bytes state_len][4 bytes img1_len][4 bytes img2_len][state_data][img1_data][img2_data]
#         """
#         # 1. 接收包含三個長度資訊的標頭 (4+4+4 = 12 bytes)
#         header = self._recv_all(12)
#         if not header:
#             return None, None, None, 0
        
#         state_len, img1_len, img2_len = struct.unpack('>III', header)
        
#         # 2. 根據標頭中的總長度，一次性接收所有剩下的資料
#         total_data_len = state_len + img1_len + img2_len
#         all_data = self._recv_all(total_data_len)
#         if not all_data:
#             return None, None, None, 0
            
#         # 3. 根據長度切割資料
#         state_data = all_data[:state_len]
#         img1_data = all_data[state_len : state_len + img1_len]
#         img2_data = all_data[state_len + img1_len :]
        
#         return state_data, img1_data, img2_data, total_data_len

#     def send_data(self, data_type, data_bytes):
#         # ... (這個函式保持不變)
#         data_type = data_type.ljust(10).encode()
#         data_len = struct.pack('>I', len(data_bytes))
#         self.conn.sendall(data_len + data_type + data_bytes)

#     def run(self):
#         last_time = time.time()
#         while True:
#             try:
#                 # --- 步驟 1: 測量接收與前處理時間 ---
#                 recv_start_time = time.time()
                
#                 # 改為呼叫新的單次接收函式
#                 state_data, img1_data, img2_data, total_bytes_received = self.recv_observation_data()

#                 if state_data is None:
#                     print("Client 端已斷開連線。")
#                     break
                
#                 recv_end_time = time.time()

#                 # --- 資料前處理 ---
#                 processing_start_time = time.time()

#                 # 解碼與轉換
#                 state = json.loads(state_data.decode())
#                 state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

#                 img1 = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_COLOR)
#                 img2 = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_COLOR)

#                 # 建立觀測 dict
#                 obs = {
#                     'observation.state': state_tensor,
#                     'observation.images.image': torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0,
#                     'observation.images.image_additional_view': torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
#                 }
                
#                 processing_end_time = time.time()

#                 # --- 步驟 2: 精確測量模型推論時間 ---
#                 inference_start_time = time.time()
#                 with torch.no_grad():
#                     action = self.policy.select_action(obs)
#                 inference_end_time = time.time()

#                 action_np = action.squeeze(0).cpu().numpy().tolist()

#                 # --- 步驟 3: 計算與輸出各種時間與效能指標 ---
#                 current_time = time.time()
#                 total_interval = current_time - last_time
#                 last_time = current_time
#                 total_freq = 1.0 / total_interval if total_interval > 0 else 0

#                 recv_interval = recv_end_time - recv_start_time
#                 processing_interval = processing_end_time - processing_start_time
#                 inference_interval = inference_end_time - inference_start_time
                
#                 transfer_speed = total_bytes_received / recv_interval if recv_interval > 0 else 0

#                 print(f"推論完成: {np.round(action_np[7],6)} | "
#                       f"總頻率: {total_freq:.2f} Hz | "
#                       f"純推論頻率: {inference_freq:.2f} Hz | "
#                       f"接收耗時: {recv_interval:.4f} s | "
#                       f"前處理耗時: {processing_interval:.4f} s | "
#                       f"傳輸速率: {transfer_speed/1024/1024:.2f} MB/s")

#                 # --- 步驟 4: 傳送推論結果回 client ---
#                 self.send_data('list', json.dumps(action_np).encode())

#             except (ConnectionResetError, BrokenPipeError) as e:
#                 print(f"❌ Client 連線中斷: {e}")
#                 break
#             except Exception as e:
#                 print(f"❌ 發生未預期錯誤: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 break

#         print("🛑 伺服器正在關閉...")
#         self.conn.close()
#         self.server_socket.close()

# if __name__ == '__main__':
#     ckpt_path = '/home/csl/lerobot/outputs/train/0721_change_action/checkpoints/last/pretrained_model'
#     server = ACTInferenceServer(ckpt_path, port=5001, device='cuda')
#     server.run()