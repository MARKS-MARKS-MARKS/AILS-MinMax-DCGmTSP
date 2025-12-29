

import os
import glob
import re
import numpy as np
import traceback # 用于捕获错误但不中断

# ================= 🔧 配置区域 =================

INPUT_DIR = r"D:\chenxvsheji\SRTP_MinMax\Path planning-6\Path planning-4\instance_initial"
OUTPUT_DIR = r"D:\chenxvsheji\SRTP_MinMax\Path planning-6\Path planning-4\instance_after"

# =================================================

class BatchGvrpConverter:
    def __init__(self, filepath, output_dir):
        self.filepath = filepath
        self.output_dir = output_dir
        self.filename = os.path.basename(filepath)
        self.name = os.path.splitext(self.filename)[0]
        
        self.nodes = [] 
        self.depots = []
        self.clusters = {} 
        self.num_nodes = 0
        self.num_vehicles = 2 
        self.final_work_distance = 0.0

    def load_tsp(self):
        """智能读取：兼容 NODE_COORD 和 DISPLAY_DATA，无法读取则返回 False"""
        if not os.path.exists(self.filepath): return False

        try:
            with open(self.filepath, 'r') as f: lines = f.readlines()
        except UnicodeDecodeError:
            print(f"⚠️ [SKIP] {self.filename}: Encoding error.")
            return False
            
        self.nodes = []
        reading_coords = False
        target_section = None
        
        if any("NODE_COORD_SECTION" in line for line in lines): target_section = "NODE_COORD_SECTION"
        elif any("DISPLAY_DATA_SECTION" in line for line in lines): target_section = "DISPLAY_DATA_SECTION"
        
        if not target_section:
            print(f"⚠️ [SKIP] {self.filename}: Pure matrix format (no coords).")
            return False

        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith("EOF"): break
            if target_section in line: reading_coords = True; continue
            
            if reading_coords:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3 and parts[0].replace('-','').isdigit():
                    try:
                        self.nodes.append([float(parts[1]), float(parts[2])])
                    except ValueError: continue

        self.nodes = np.array(self.nodes)
        self.num_nodes = len(self.nodes)
        
        if self.num_nodes < 3:
            print(f"⚠️ [SKIP] {self.filename}: Too few nodes ({self.num_nodes}).")
            return False
        return True

    def configure_vehicles(self):
        """智能设置激光头数量"""
        if self.num_nodes < 100: self.num_vehicles = 2
        elif self.num_nodes < 500: self.num_vehicles = 4
        elif self.num_nodes < 1000: self.num_vehicles = 6
        elif self.num_nodes < 5000: self.num_vehicles = 8
        else: self.num_vehicles = 16 

    def generate_clusters_adaptive(self):
        """自适应分簇：确保零件数符合要求"""
        min_groups = max(2, int(self.num_nodes / 6))
        
        # 初始目标：1/5
        current_target_cells = int(self.num_nodes / 5)
        step_size = max(1, int(self.num_nodes / 10))
        
        for attempt in range(20): # 最多重试20次
            self._grid_partition_fast(current_target_cells) # 使用极速版算法
            actual_groups = len(self.clusters)
            
            if actual_groups >= min_groups: break
            current_target_cells += step_size

    def _grid_partition_fast(self, target_cells):
        """【极速优化版】O(N) 复杂度的网格划分"""
        if target_cells < 1: target_cells = 1
        grid_dim = int(np.sqrt(target_cells))
        if grid_dim < 1: grid_dim = 1
        
        min_x, min_y = np.min(self.nodes, axis=0)
        max_x, max_y = np.max(self.nodes, axis=0)
        
        # 防止分母为0
        width = max_x - min_x
        height = max_y - min_y
        if width == 0: width = 1.0
        if height == 0: height = 1.0
        
        # 增加微小偏移防止边界溢出
        width *= 1.0001
        height *= 1.0001
        
        cell_width = width / grid_dim
        cell_height = height / grid_dim
        
        self.clusters = {}
        
        # 直接计算每个点属于哪个格子 (O(N) 速度)
        # col = (x - min_x) // cell_width
        # row = (y - min_y) // cell_height
        
        # 批量计算以提升 Python 速度
        cols = ((self.nodes[:, 0] - min_x) // cell_width).astype(int)
        rows = ((self.nodes[:, 1] - min_y) // cell_height).astype(int)
        
        # 这里的 key 是 (row, col) 元组
        temp_clusters = {}
        
        for idx in range(self.num_nodes):
            r, c = rows[idx], cols[idx]
            key = (r, c)
            if key not in temp_clusters:
                temp_clusters[key] = []
            temp_clusters[key].append(idx + 1) # 1-based ID
            
        # 重新编号 Group ID 为 1, 2, 3...
        for i, key in enumerate(temp_clusters.keys()):
            self.clusters[i + 1] = temp_clusters[key]

    def generate_depots(self):
        min_x, min_y = np.min(self.nodes, axis=0)
        max_x, max_y = np.max(self.nodes, axis=0)
        w, h = max_x - min_x, max_y - min_y
        margin = max(w, h) * 0.1
        bounds = {'l': min_x - margin, 'r': max_x + margin, 'b': min_y - margin, 't': max_y + margin}
        
        self.depots = []
        if self.num_vehicles <= 4:
            corners = [[bounds['l'], bounds['b']], [bounds['r'], bounds['t']], [bounds['l'], bounds['t']], [bounds['r'], bounds['b']]]
            for i in range(self.num_vehicles): self.depots.append(corners[i % 4])
        else:
            top_n = self.num_vehicles // 2
            bot_n = self.num_vehicles - top_n
            for i in range(bot_n):
                x = bounds['l'] + (bounds['r'] - bounds['l']) * (i / max(1, bot_n - 1))
                self.depots.append([x, bounds['b']])
            for i in range(top_n):
                x = bounds['l'] + (bounds['r'] - bounds['l']) * (i / max(1, top_n - 1))
                self.depots.append([x, bounds['t']])

    def write_gvrp(self):
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        output_path = os.path.join(self.output_dir, f"{self.name}_GVRP_Spatial.txt")
        
        min_x, min_y = np.min(self.nodes, axis=0)
        max_x, max_y = np.max(self.nodes, axis=0)
        diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        self.final_work_distance = diagonal * 0.7
        
        with open(output_path, 'w') as f:
            f.write(f"NAME: {self.name}-GVRP-Spatial\n")
            f.write(f"TYPE: GVRP\n")
            f.write(f"DIMENSION: {self.num_nodes}\n")
            f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write(f"VEHICLES: {self.num_vehicles}\n")
            f.write(f"CAPACITY: 200\n")
            f.write(f"WORK_DISTANCE: {self.final_work_distance:.4f}\n")
            f.write(f"NUM_OF_GROUPS: {len(self.clusters)}\n")
            
            f.write("DEPOT_COORD_SECTION\n")
            for i, (x, y) in enumerate(self.depots): f.write(f"{i+1}\t{x:.4f}\t{y:.4f}\n")
            
            f.write("NODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(self.nodes): f.write(f"{i+1}\t{x:.4f}\t{y:.4f}\n")
            
            f.write("MUTUALLY_EXCLUSIVE_GROUP_SECTION\n")
            for gid in sorted(self.clusters.keys()):
                f.write(f"{gid}\t" + "\t".join(map(str, self.clusters[gid])) + "\n")
        
        print(f"✅ [Done] {self.name:<15} -> Nodes: {self.num_nodes:<5} Groups: {len(self.clusters):<3} Vehicles: {self.num_vehicles}")

def main():
    print("🚀 Starting Batch Conversion...\n")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    tsp_files = glob.glob(os.path.join(INPUT_DIR, "*.tsp"))
    
    success, failed = 0, 0
    
    for file_path in tsp_files:
        # ==========================================
        # 核心修改：全局 Try-Catch 保护
        # 任何算例报错，直接跳过，绝不卡死
        # ==========================================
        try:
            converter = BatchGvrpConverter(file_path, OUTPUT_DIR)
            if not converter.load_tsp():
                failed += 1
                continue # 正常跳过（如矩阵文件）
            
            converter.configure_vehicles()
            converter.generate_clusters_adaptive()
            converter.generate_depots()
            converter.write_gvrp()
            success += 1
            
        except Exception as e:
            print(f"❌ [CRITICAL ERROR] Failed to convert {os.path.basename(file_path)}")
            # 打印详细错误信息但继续运行
            # traceback.print_exc() 
            failed += 1
            continue

    print(f"\n🎉 Process Complete! Success: {success}, Skipped/Failed: {failed}")

if __name__ == "__main__":
    main()