import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import threading
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.ndimage import maximum_filter
import matplotlib.lines  # 添加这一行确保matplotlib.lines可用
from sklearn.linear_model import LinearRegression  # 添加线性回归模型导入
from matplotlib.widgets import RectangleSelector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

try:
    import umap
except ImportError:
    umap = None
import tkinter.simpledialog
from scipy.stats import wasserstein_distance  # 确保导入

try:
    from xgboost import XGBRegressor
except Exception:
    # 在打包环境中，可能抛出 XGBoostLibraryNotFound（非 ImportError），这里统一降级
    XGBRegressor = None

# 尝试导入 Pillow 以显示 jpg logo（若不可用则优雅降级）
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 全局暗色科技风格 Matplotlib 配置
# 切换为浅色科技风格 Matplotlib 配置（非纯白）
plt.rcParams.update({
    'figure.facecolor': '#f5f7fb',
    'axes.facecolor': '#f7f9fc',
    'axes.edgecolor': '#1f2937',
    'text.color': '#111827',
    'axes.labelcolor': '#111827',
    'xtick.color': '#374151',
    'ytick.color': '#374151',
    'grid.color': '#e5e7ef'
})

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("FAIR: Feature-space Analysis and Insight for Reliability")
        self.root.geometry("1200x800")

        # 应用浅色科技风格主题
        self._apply_light_theme()

        # 使主窗口可调整大小
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # 顶部导航栏（浅色）
        self.navbar = tk.Frame(self.main_frame, height=52, bg='#eef2f9')
        self.navbar.grid(row=0, column=0, sticky="ew")
        self.navbar.grid_propagate(False)

        # 内容区域
        self.content = ttk.Frame(self.main_frame)
        self.content.grid(row=1, column=0, sticky="nsew")

        # 配置主框架的权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # 存储数据
        self.data = None
        self.tsne_result = None
        self.center_point = None
        self.tsne_model = None
        self.pca_model = None
        self.umap_model = None
        self.scaler = None
        
        # 创建页面
        self.pages = {}
        self.current_page = None

        # 构建顶部导航栏按钮
        self._build_navbar()
        
        # 初始化页面
        self.create_pages()
        
        # 默认显示欢迎页面
        self.show_page("page0")
    
    def create_pages(self):
        # 创建欢迎页 Page0
        self.pages["page0"] = Page0(self.content, self)
        self.pages["page0"].grid(row=0, column=0, sticky="nsew")

        # 创建页面1 (原有的T-SNE分析页面)
        self.pages["page1"] = Page1(self.content, self)
        self.pages["page1"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面3 (样本差异分析页面)
        self.pages["page3"] = Page3(self.content, self)
        self.pages["page3"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面4 (特征分布分析页面)
        self.pages["page4"] = Page4(self.content, self)
        self.pages["page4"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面5 (主动采样)
        self.pages["page5"] = Page5(self.content, self)
        self.pages["page5"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面6 (主动采样2)
        self.pages["page6"] = Page6(self.content, self)
        self.pages["page6"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面7 (性能曲线)
        self.pages["page7"] = Page7(self.content, self)
        self.pages["page7"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面8 (交互式PCA)
        self.pages["page8"] = Page8(self.content, self)
        self.pages["page8"].grid(row=0, column=0, sticky="nsew")
        
        # 创建页面9 (增量学习)
        self.pages["page9"] = Page9(self.content, self)
        self.pages["page9"].grid(row=0, column=0, sticky="nsew")
        
        # 配置content的网格权重
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        
        # 初始时隐藏所有页面
        for page in self.pages.values():
            page.grid_remove()
    
    def show_page(self, page_name):
        # 隐藏当前页面
        if self.current_page:
            self.current_page.grid_remove()
        
        # 显示选定的页面
        self.current_page = self.pages[page_name]
        self.current_page.grid()
        # 更新导航栏高亮
        if hasattr(self, '_set_active_nav'):
            self._set_active_nav(page_name)
        
        # 触发页面更新
        self.current_page.event_generate("<<Visibility>>")
    
    def update_data(self, data, tsne_result=None):
        """更新共享数据"""
        self.data = data
        self.tsne_result = tsne_result

    def _apply_light_theme(self):
        """应用 ttk 浅色科技风格主题（非纯白）"""
        try:
            style = ttk.Style()
            # 使用稳定主题作为基底
            try:
                style.theme_use('clam')
            except Exception:
                pass

            # 根窗口背景（非纯白）
            self.root.configure(bg='#f5f7fb')

            # 基础控件
            style.configure('TFrame', background='#f5f7fb')
            style.configure('TLabel', background='#f5f7fb', foreground='#1f2937')
            style.configure('TLabelframe', background='#f5f7fb', foreground='#1f2937')
            style.configure('TLabelframe.Label', background='#f5f7fb', foreground='#6b7280')
            style.configure('TButton', background='#e5e7eb', foreground='#111827', padding=(10, 6))
            style.map('TButton', background=[('active', '#d1d5db')])

            # 顶部导航按钮样式
            style.configure('Nav.TButton', background='#e2e8f0', foreground='#111827', padding=(14, 8), borderwidth=0, font=('Arial', 10, 'bold'))
            style.map('Nav.TButton', background=[('active', '#cfd8e3')])
            style.configure('NavActive.TButton', background='#2563eb', foreground='#ffffff', padding=(14, 8), font=('Arial', 10, 'bold'))

            # Treeview（用于表格）
            style.configure('Treeview', background='#ffffff', foreground='#111827', fieldbackground='#ffffff')
            style.configure('Treeview.Heading', background='#e5e7eb', foreground='#111827')
        except Exception:
            pass

    def _build_navbar(self):
        """构建顶部导航栏按钮"""
        self.nav_buttons = {}
        btns = [
            ("Welcome", "page0"),
            ("Global Visualization", "page1"),
            ("Comparative Distribution Plot", "page3"),
            ("Split‑Violin Diagnosis", "page4"),
            ("Exploration based Sampling", "page5"),
            ("Redundancy reduced Sampling", "page6"),
            ("Sampling Efficiency", "page7"),
            ("Augmented Data PCA", "page8"),
            ("Exploration Impact", "page9"),
        ]

        # 使用网格布局让按钮自适应宽度
        for i, (text, name) in enumerate(btns):
            btn = ttk.Button(self.navbar, text=text, style='Nav.TButton', command=lambda n=name: self.show_page(n))
            btn.grid(row=0, column=i, padx=6, pady=6, sticky='ns')
            self.navbar.grid_columnconfigure(i, weight=1)
            self.nav_buttons[name] = btn

        # 占位扩展
        self.navbar.grid_columnconfigure(len(btns), weight=10)

    def _set_active_nav(self, page_name):
        for name, btn in self.nav_buttons.items():
            if name == page_name:
                btn.configure(style='NavActive.TButton')
            else:
                btn.configure(style='Nav.TButton')


class Page0(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # 容器
        container = ttk.Frame(self)
        container.pack(expand=True, fill='both')

        # 背景装饰画布（浅色科技氛围）
        self.bg_canvas = tk.Canvas(container, highlightthickness=0, bd=0, bg='#f5f7fb')
        self.bg_canvas.pack(expand=True, fill='both')
        container.bind('<Configure>', lambda e: self._draw_bg(e.width, e.height))

        # 居中容器，将内容严格居中
        center_frame = ttk.Frame(container)
        center_frame.place(relx=0.5, rely=0.5, anchor='center')
        center_frame.lift()

        # LOGO 区域（放大尺寸）
        logo_frame = ttk.Frame(center_frame)
        logo_frame.pack(pady=10)

        logo_loaded = False
        logo_path_cand = [
            os.path.join(os.getcwd(), 'logo.jpg'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.jpg')
        ]
        logo_path = next((p for p in logo_path_cand if os.path.exists(p)), None)

        if Image is not None and ImageTk is not None and logo_path is not None:
            try:
                img = Image.open(logo_path)
                # 约束最大尺寸（放大）
                max_w, max_h = 520, 520
                ratio = min(max_w / img.width, max_h / img.height, 1.0)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                resampling = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                img = img.resize(new_size, resampling)
                self.tk_img = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(logo_frame, image=self.tk_img)
                logo_label.pack()
                logo_loaded = True
            except Exception:
                logo_loaded = False

        if not logo_loaded:
            # 回退文本 Logo
            title = ttk.Label(logo_frame, text="FAIR", font=("Arial", 42, "bold"))
            subtitle = ttk.Label(logo_frame, text="Feature‑space Analysis and Insight for Reliability", font=("Arial", 16))
            title.pack()
            subtitle.pack()

        # 标题与说明
        text_frame = ttk.Frame(center_frame)
        text_frame.pack(pady=10)
        title2 = ttk.Label(text_frame, text="Welcome to FAIR Visualization Suite", font=("Arial", 24, "bold"))
        desc = ttk.Label(text_frame, text="An integrated tool for exploration, diagnosis, and sampling in high‑dimensional feature spaces", font=("Arial", 13))
        title2.pack(pady=(10, 2))
        desc.pack()

        # 主题色点缀条
        accent = ttk.Frame(text_frame)
        accent.pack(pady=(8, 0))
        bar1 = tk.Frame(accent, width=70, height=4, bg='#2563eb')
        bar2 = tk.Frame(accent, width=70, height=4, bg='#38bdf8')
        bar3 = tk.Frame(accent, width=70, height=4, bg='#a78bfa')
        bar1.pack(side='left', padx=6)
        bar2.pack(side='left', padx=6)
        bar3.pack(side='left', padx=6)

        # 行为按钮
        action_frame = ttk.Frame(center_frame)
        action_frame.pack(pady=20)
        start_btn = ttk.Button(action_frame, text="Get Started", style='NavActive.TButton', command=lambda: controller.show_page('page1'))
        start_btn.pack(ipadx=18, ipady=8)

    def _draw_bg(self, w: int, h: int):
        """绘制欢迎页背景装饰（柔和几何光斑与连接线）"""
        try:
            c = self.bg_canvas
            c.delete('all')
            c.configure(bg='#f5f7fb')

            # 柔和光斑（大圆）
            c.create_oval(-0.18*w, -0.25*h, 0.30*w, 0.23*h, fill='#e9f1ff', outline='')
            c.create_oval(0.70*w, 0.60*h, 1.15*w, 1.05*h, fill='#ecf7ff', outline='')
            c.create_oval(0.65*w, -0.10*h, 1.05*w, 0.30*h, fill='#f2ecff', outline='')

            # 连接线网络（轻微对比）
            line_color = '#d3dbe8'
            # 预设若干星座风格的节点（分散在左下与右上区域，并少量中部）
            node_fracs = [
                # 左下簇
                (0.12, 0.68), (0.18, 0.74), (0.24, 0.70), (0.20, 0.80), (0.30, 0.76), (0.28, 0.66),
                # 右上簇
                (0.74, 0.30), (0.76, 0.22), (0.82, 0.18), (0.88, 0.26), (0.84, 0.34), (0.78, 0.36),
                # 过渡点
                (0.55, 0.22), (0.60, 0.28), (0.40, 0.62)
            ]
            nodes = [(fx*w, fy*h) for fx, fy in node_fracs]

            # 使用并查集构造无回路的连线（森林），提升密度但避免三角回路
            import math
            n = len(nodes)
            parent = list(range(n))
            degree = [0] * n

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False
                parent[rb] = ra
                return True

            max_dist = 0.16 * w  # 限制连接距离，避免跨区连线
            for i in range(n):
                xi, yi = nodes[i]
                candidates = []
                for j in range(i):
                    xj, yj = nodes[j]
                    d = math.hypot(xi - xj, yi - yj)
                    if d <= max_dist:
                        candidates.append((d, j))
                candidates.sort(key=lambda t: t[0])
                connected = 0
                for _, j in candidates:
                    if connected >= 2:
                        break
                    if degree[i] >= 3 or degree[j] >= 3:
                        continue
                    if union(i, j):
                        xj, yj = nodes[j]
                        c.create_line(xi, yi, xj, yj, fill=line_color, width=1)
                        degree[i] += 1
                        degree[j] += 1
                        connected += 1

            for (x, y) in nodes:
                c.create_oval(x-3, y-3, x+3, y+3, fill='#c7d2fe', outline='')
        except Exception:
            pass

        # 移除错位装饰线，保持界面简洁

class Page1(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # 创建控制面板
        self.control_frame = ttk.LabelFrame(self, text="Global Visualization Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=5, pady=5)
        btn_width = 20
        # 添加数据加载按钮
        ttk.Button(self.control_frame, text="Load Training Set", width=btn_width, command=self.load_data).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        # 添加降维方法选择
        ttk.Label(self.control_frame, text="Dimensionality Reduction:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.dim_reduction_var = tk.StringVar(value="t-SNE")
        dim_reduction_combo = ttk.Combobox(self.control_frame, textvariable=self.dim_reduction_var, values=["t-SNE", "PCA", "UMAP"], state="readonly", width=btn_width-2)
        dim_reduction_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        # t-SNE参数设置
        ttk.Label(self.control_frame, text="perplexity:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.perplexity_var = tk.StringVar(value="30")
        ttk.Entry(self.control_frame, textvariable=self.perplexity_var, width=btn_width-2).grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        # 运行按钮
        ttk.Button(self.control_frame, text="Run Global Projection", width=btn_width, command=self.run_analysis).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        # 导出图片按钮
        export_button = ttk.Button(self.control_frame, text="Export Image", width=btn_width, command=self.export_images)
        export_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        # 创建图表占位框架
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        # 配置网格权重
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        # 初始化变量
        self.canvas = None
        self.annotation = None
        self.center_point = None
        self.background = None
        self.tree = None
    
    def load_data(self):
        """加载数据文件（彻底不检测MAE/MSE）"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # 创建进度条窗口
                progress_window = tk.Toplevel(self)
                progress_window.title("Loading Data")
                progress_window.geometry("300x150")
                progress_window.transient(self)  # 设置为主窗口的子窗口
                # 居中显示
                progress_window.geometry("+%d+%d" % (
                    self.winfo_rootx() + 50,
                    self.winfo_rooty() + 50))
                # 添加进度条和标签
                label = ttk.Label(progress_window, text="Loading data...")
                label.pack(pady=10)
                progress_bar = ttk.Progressbar(
                    progress_window, 
                    length=200, 
                    mode='determinate'
                )
                progress_bar.pack(pady=10)
                # 获取文件大小
                file_size = os.path.getsize(file_path)
                chunk_size = file_size // 100  # 将文件分成100份
                # 先读取文件头以获取列名
                header = pd.read_csv(file_path, nrows=0)
                columns = header.columns
                # 分块读取文件
                chunks = []
                bytes_read = 0
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    bytes_read += chunk.memory_usage(deep=True).sum()
                    progress = min(100, int((bytes_read / file_size) * 100))
                    progress_bar['value'] = progress
                    label['text'] = f"Loading data... {progress}%"
                    progress_window.update()
                # 合并所有数据块
                data = pd.concat(chunks, ignore_index=True)
                # 获取特征列（只排除y）
                feature_cols = [col for col in data.columns if col.lower() != 'y']
                # 处理缺失值
                missing_info = []
                for col in data.columns:
                    missing_count = data[col].isnull().sum()
                    if missing_count > 0:
                        # 使用平均值填充数值型列
                        if pd.api.types.is_numeric_dtype(data[col]):
                            mean_val = data[col].mean()
                            data[col].fillna(mean_val, inplace=True)
                            missing_info.append(f"{col}: {missing_count} missing values filled with mean {mean_val:.4f}")
                        else:
                            # 对于非数值型列，使用众数填充
                            mode_val = data[col].mode()[0]
                            data[col].fillna(mode_val, inplace=True)
                            missing_info.append(f"{col}: {missing_count} missing values filled with mode '{mode_val}'")
                self.controller.data = data
                # 关闭进度条窗口
                progress_window.destroy()
                # 显示数据加载和处理信息
                info_message = f"Loaded data, total {len(data)} rows\nNumber of features: {len(feature_cols)}"
                if missing_info:
                    info_message += "\n\nMissing values handled:"
                    info_message += "\n".join(missing_info)
                tk.messagebox.showinfo("Success", info_message)
            except Exception as e:
                if 'progress_window' in locals():
                    progress_window.destroy()
                tk.messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def create_plot_area(self):
        # 如果已存在画布，先清除
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        # 获取plot_frame的大小
        self.plot_frame.update()
        width = self.plot_frame.winfo_width() / 100
        height = self.plot_frame.winfo_height() / 100
        # 创建图表区域（只主图和colorbar）
        self.fig = plt.figure(figsize=(width, height))
        self.ax_main = self.fig.add_axes([0.08, 0.12, 0.87, 0.80])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.plot_frame.bind('<Configure>', self.on_resize)
    
    def run_analysis(self):
        """运行T-SNE分析"""
        if self.controller.data is None:
            tk.messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            # 创建进度条窗口
            progress_window = tk.Toplevel(self)
            progress_window.title("Analysis Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)  # 设置为主窗口的子窗口
            progress_window.grab_set()  # 设置为模态窗口
            
            # 禁用关闭按钮
            progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # 居中显示
            progress_window.geometry("+%d+%d" % (
                self.winfo_rootx() + 50,
                self.winfo_rooty() + 50))
            
            # 添加进度条和标签
            label = ttk.Label(progress_window, text="Performing dimensionality reduction analysis...")
            label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(
                progress_window, 
                length=200, 
                mode='indeterminate'
            )
            progress_bar.pack(pady=10)
            
            # 开始进度条动画
            progress_bar.start(10)
            progress_window.update()
            
            # 获取特征列
            feature_cols = [col for col in self.controller.data.columns 
                          if col not in ['MAE', 'MSE']]
            X = self.controller.data[feature_cols]
            
            # 根据选择的降维方法执行降维
            if self.dim_reduction_var.get() == "t-SNE":
                # 检查perplexity值
                perplexity = float(self.perplexity_var.get())
                if perplexity >= len(self.controller.data):
                    progress_window.destroy()
                    tk.messagebox.showerror("Error", 
                        f"Perplexity value ({perplexity}) cannot be greater than or equal to the number of samples ({len(self.controller.data)}).\nIt is recommended to set it to 1/3 or less of the sample size.")
                    return
                
                # 更新标签
                self.after_idle(lambda: label.config(
                    text="Running t-SNE dimensionality reduction...\nThis may take several minutes"))
                
                # 定义计算函数
                def compute_tsne():
                    try:
                        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        result = tsne.fit_transform(X)
                        # 使用after_idle在主线程中更新UI
                        self.after_idle(lambda: complete_analysis(result, tsne))
                    except Exception as e:
                        error_msg = str(e)
                        self.after_idle(lambda error_msg=error_msg: handle_error(error_msg))
                
                def complete_analysis(result, model):
                    self.controller.tsne_result = result
                    self.controller.tsne_model = model
                    progress_window.destroy()
                    self.create_plot_area()
                    self.plot_results()
                
                def handle_error(error_msg):
                    progress_window.destroy()
                    tk.messagebox.showerror("Error", f"t-SNE analysis failed: {error_msg}")
                
                # 在新线程中运行计算
                thread = threading.Thread(target=compute_tsne)
                thread.daemon = True
                thread.start()
                
            elif self.dim_reduction_var.get() == "PCA":
                self.after_idle(lambda: label.config(text="Running PCA dimensionality reduction..."))
                
                def compute_pca():
                    try:
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA
                        
                        # 标准化数据
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # 运行PCA
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X_scaled)
                        
                        # 使用after_idle在主线程中更新UI
                        self.after_idle(lambda: complete_analysis(pca_result, pca, scaler))
                    except Exception as e:
                        error_msg = str(e)
                        self.after_idle(lambda error_msg=error_msg: handle_error(error_msg))
                
                def complete_analysis(result, model, scaler):
                    self.controller.tsne_result = result
                    self.controller.pca_model = model
                    self.controller.scaler = scaler
                    progress_window.destroy()
                    self.create_plot_area()
                    self.plot_results()
                
                def handle_error(error_msg):
                    progress_window.destroy()
                    tk.messagebox.showerror("Error", f"PCA analysis failed: {error_msg}")
                
                # 在新线程中运行计算
                thread = threading.Thread(target=compute_pca)
                thread.daemon = True
                thread.start()
            elif self.dim_reduction_var.get() == "UMAP":
                self.after_idle(lambda: label.config(text="Running UMAP dimensionality reduction..."))
                
                def compute_umap():
                    try:
                        from sklearn.preprocessing import StandardScaler
                        if umap is None:
                            raise ImportError("umap-learn library not installed, please install it first!")
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        reducer = umap.UMAP(n_components=2, random_state=42)
                        umap_result = reducer.fit_transform(X_scaled)
                        self.after_idle(lambda: complete_analysis(umap_result, reducer, scaler))
                    except Exception as exc:
                        err_msg = str(exc)
                        self.after_idle(lambda err_msg=err_msg: handle_error(err_msg))
                
                def complete_analysis(result, model, scaler):
                    self.controller.tsne_result = result
                    self.controller.umap_model = model
                    self.controller.scaler = scaler
                    progress_window.destroy()
                    self.create_plot_area()
                    self.plot_results()
                
                def handle_error(error_msg):
                    progress_window.destroy()
                    tk.messagebox.showerror("Error", f"UMAP analysis failed: {error_msg}")
                
                # 在新线程中运行计算
                thread = threading.Thread(target=compute_umap)
                thread.daemon = True
                thread.start()
                return
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            tk.messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
    
    def plot_results(self):
        """Page1的绘图方法（重构：所有点绿色半透明，无MAE）"""
        try:
            plt.rcParams.update({
                'font.size': 15,
                'axes.titlesize': 15,
                'axes.labelsize': 15,
                'xtick.labelsize': 15,
                'ytick.labelsize': 15
            })
            self.ax_main.clear()
            feature_cols = [col for col in self.controller.data.columns if col.lower() != 'y']
            tsne_data = self.controller.tsne_result.copy()
            valid_mask = np.isfinite(tsne_data).all(axis=1)
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                print(f"Warning: {invalid_count} invalid values found in TSNE results (NaN or Inf), these points will be filtered out")
                tsne_data = tsne_data[valid_mask]
                if len(tsne_data) < 10:
                    tk.messagebox.showerror("Error", "Too few valid data points for analysis")
                    return
            # 只绘制绿色点，不再计算/显示中心点
            self.sc = self.ax_main.scatter(self.controller.tsne_result[:, 0], self.controller.tsne_result[:, 1], color='#90EE90', alpha=0.7, s=50, edgecolors='none')
            x_min, x_max = self.controller.tsne_result[:, 0].min(), self.controller.tsne_result[:, 0].max()
            y_min, y_max = self.controller.tsne_result[:, 1].min(), self.controller.tsne_result[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            self.ax_main.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            self.ax_main.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.ax_main.tick_params(axis='both', which='major', labelsize=15, pad=8)
            self.fig.suptitle("Global Feature-space Visualization", y=0.95, fontsize=15)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error during plotting: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_resize(self, event):
        # 当窗口大小改变时，使用防抖动机制
        if self.controller.data is not None:  # 只在有数据时处理
            # 如果之前的定时器存在，取消它
            if hasattr(self, 'resize_timer') and self.resize_timer is not None:
                self.after_cancel(self.resize_timer)
            
            # 设置新的定时器
            self.resize_timer = self.after(100, self.delayed_resize)
    
    def delayed_resize(self):
        # 实际执行重绘的函数
        self.create_plot_area()
        self.plot_results()
        # 重置定时器
        self.resize_timer = None
    
    def on_mouse_move(self, event):
        # 检查事件是否在主绘图区域内
        if not event.inaxes or event.inaxes != self.ax_main:
            if hasattr(self, 'annotation') and self.annotation:
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
            return
        if not hasattr(self, 'tree') or self.tree is None:
            if hasattr(self.controller, 'tsne_result') and self.controller.tsne_result is not None:
                from scipy.spatial import cKDTree
                self.tree = cKDTree(self.controller.tsne_result)
            else:
                return
        try:
            dist, idx = self.tree.query([event.xdata, event.ydata], k=1)
            threshold = 0.5
            if dist > threshold:
                if hasattr(self, 'annotation') and self.annotation:
                    self.annotation.set_visible(False)
                    self.canvas.draw_idle()
                return
            features = self.controller.data.iloc[idx]
            feature_cols = [col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]
            text = ""
            for i, col in enumerate(feature_cols[:3]):
                text += f"{col}: {features[col]:.4f}\n"
            if hasattr(self, 'annotation') and self.annotation:
                self.annotation.set_visible(True)
                self.annotation.xy = (self.controller.tsne_result[idx, 0], self.controller.tsne_result[idx, 1])
                self.annotation.set_text(text)
            else:
                self.annotation = self.ax_main.annotate(
                    text,
                    xy=(self.controller.tsne_result[idx, 0], self.controller.tsne_result[idx, 1]),
                    xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", 
                             fc="yellow", 
                             alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )
            self.canvas.draw_idle()
        except Exception as e:
            import traceback
            print(f"Mouse move event error: {str(e)}")
            traceback.print_exc()

    def on_draw(self, event):
        """重新缓存背景"""
        self.background = self.canvas.copy_from_bbox(self.ax_main.bbox)

    def export_images(self, filename="exported_image.png"):
        """导出当前页面的所有图片为高分辨率图像"""
        try:
            # 保存当前图像
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            tk.messagebox.showinfo("Export Success", f"Image successfully exported as {filename}")
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Error exporting image: {str(e)}")

class Page3(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # 创建控制面板
        self.control_frame = ttk.LabelFrame(self, text="Comparative Distribution Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        
        # 设置统一的按钮宽度
        btn_width = 20
        
        # 添加数据加载按钮
        ttk.Button(self.control_frame, text="Load Test Data", 
                  width=btn_width,
                  command=self.load_test_data).grid(row=0, column=0, padx=5, pady=5)
        
        # 添加运行按钮
        ttk.Button(self.control_frame, text="Run Comparative Plot", 
                  width=btn_width,
                  command=self.run_analysis).grid(row=1, column=0, padx=5, pady=5)
        
        # 添加显示/隐藏训练集点的按钮
        self.show_train = tk.BooleanVar(value=True)
        self.toggle_button = ttk.Button(self.control_frame, 
                                      text="Hide Training Set",
                                      width=btn_width,
                                      command=self.toggle_train_points)
        self.toggle_button.grid(row=2, column=0, padx=5, pady=5)
        
        # 添加导出按钮
        ttk.Button(self.control_frame, text="Export Figure", 
                  width=btn_width,
                  command=self.export_images).grid(row=3, column=0, padx=5, pady=5)
        
        # 创建图表区域
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 配置网格权重
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        # 初始化变量
        self.test_data = None
        self.test_result = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.annotation = None
        self.background = None  # 添加背景缓存变量
        self.tree = None       # 添加KD树变量
        
        # 绑定显示事件
        self.bind("<<Visibility>>", self.on_visibility)
        
        # 初始化图形对象
        self.create_plot_area()
    
    def create_plot_area(self):
        """创建绘图区域"""
        # 如果已存在画布，先清除
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        
        # 创建新的图形对象
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_plot(self):
        """更新图表（重构：新导入点统一蓝色，无Page2曲线，无中心点）"""
        if self.test_result is None:
            tk.messagebox.showerror("Error", "Dimensionality reduction result is empty. Please check the analysis process or parameters.")
            return
        try:
            self.ax.clear()
            self.train_scatter = self.ax.scatter(self.controller.tsne_result[:, 0], 
                                        self.controller.tsne_result[:, 1], 
                                        color='#90EE90',  # 浅绿色
                                        alpha=0.3,        # 更透明
                                        s=100,
                                        label='Training Set')
            # 不再绘制中心点
            self.test_scatter = self.ax.scatter(self.test_result[:, 0], 
                                           self.test_result[:, 1],
                                              color='#007bff',
                                           alpha=0.7,
                                           s=80,
                                           label='Test Set')
            self.tree = cKDTree(self.test_result)
            self.ax.set_xlabel("Dimension 1")
            self.ax.set_ylabel("Dimension 2")
            self.ax.set_title("Comparative Distribution Plot")
            self.train_scatter.set_visible(self.show_train.get())
            self.ax.legend(loc='upper right', 
                        bbox_to_anchor=(0.98, 0.98),
                        framealpha=0.8,
                        title="Sample Types")
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to update plot: {str(e)}")
    
    def load_test_data(self):
        """加载测试集数据"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                # 获取训练集的特征列（不包括MAE和MSE）
                train_feature_cols = [col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]
                # 检查测试集是否包含所有必需的特征列
                missing_cols = [col for col in train_feature_cols if col not in self.test_data.columns]
                if missing_cols:
                    raise ValueError(f"Test set is missing the following features: {', '.join(missing_cols)}")
                # 处理缺失值
                missing_info = []
                for col in self.test_data.columns:
                    missing_count = self.test_data[col].isnull().sum()
                    if missing_count > 0:
                        # 使用平均值填充数值型列
                        if pd.api.types.is_numeric_dtype(self.test_data[col]):
                            mean_val = self.test_data[col].mean()
                            self.test_data[col].fillna(mean_val, inplace=True)
                            missing_info.append(f"{col}: {missing_count} missing values filled with mean {mean_val:.4f}")
                        else:
                            # 对于非数值型列，使用众数填充
                            mode_val = self.test_data[col].mode()[0]
                            self.test_data[col].fillna(mode_val, inplace=True)
                            missing_info.append(f"{col}: {missing_count} missing values filled with mode '{mode_val}'")
                # 如果测试集包含额外的列（如MAE、MSE），这是允许的
                info_message = f"Loaded test data, total {len(self.test_data)} rows"
                if missing_info:
                    info_message += "\n\nMissing values handled:"
                    info_message += "\n".join(missing_info)
                tk.messagebox.showinfo("Success", info_message)
            except Exception as e:
                tk.messagebox.showerror("错误", f"加载数据失败: {str(e)}")
                self.test_data = None
    
    def run_analysis(self):
        """运行分析"""
        if (self.controller.data is None or 
            self.controller.tsne_result is None):
            tk.messagebox.showerror("Error", "Please complete training set analysis in Page 1 first")
            return
        if self.test_data is None:
            tk.messagebox.showerror("Error", "Please load test data first")
            return
        try:
            # 创建进度条窗口
            progress_window = tk.Toplevel(self)
            progress_window.title("Analysis Progress")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            progress_window.grab_set()
            # 禁用关闭按钮
            progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
            # 居中显示
            progress_window.geometry("+%d+%d" % (
                self.winfo_rootx() + 50,
                self.winfo_rooty() + 50))
            # 添加进度条和标签
            label = ttk.Label(progress_window, text="Performing dimensionality reduction...")
            label.pack(pady=10)
            progress_bar = ttk.Progressbar(
                progress_window, 
                length=200, 
                mode='indeterminate'
            )
            progress_bar.pack(pady=10)
            # 开始进度条动画
            progress_bar.start(10)
            progress_window.update()
            # 获取特征列（不包括MAE和MSE）
            feature_cols = [col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]
            X_test = self.test_data[feature_cols].values
            # 获取页面1的降维方法
            page1 = self.controller.pages["page1"]
            reduction_method = page1.dim_reduction_var.get()
            # 根据降维方法进行处理
            if reduction_method == "t-SNE":
                # 对于t-SNE，我们需要使用与训练集相同的参数重新拟合
                if hasattr(self.controller, 'tsne_model') and self.controller.tsne_model is not None:
                    # 获取原始t-SNE的参数
                    perplexity = self.controller.tsne_model.perplexity
                    # 创建新的t-SNE模型，使用相同的参数
                    tsne = TSNE(n_components=2, 
                               perplexity=min(perplexity, len(X_test) - 1),  # 确保perplexity小于样本数
                               random_state=42)
                    # 对测试集进行降维
                    self.test_result = tsne.fit_transform(X_test)
                else:
                    raise ValueError("t-SNE model not found, please complete training set analysis in Page 1 first")
            elif reduction_method == "PCA":
                # 使用已有的PCA模型进行转换
                if hasattr(self.controller, 'scaler') and hasattr(self.controller, 'pca_model'):
                    X_test_scaled = self.controller.scaler.transform(X_test)
                    self.test_result = self.controller.pca_model.transform(X_test_scaled)
                else:
                    raise ValueError("PCA model not found, please complete training set analysis in Page 1 first")
            elif reduction_method == "UMAP":
                # 新增UMAP分支
                if not (hasattr(self.controller, 'umap_model') and self.controller.umap_model is not None and hasattr(self.controller, 'scaler') and self.controller.scaler is not None):
                    raise ValueError("UMAP模型未初始化，请先在Page1用UMAP方法完成训练集降维分析！")
                feature_cols = [col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]
                X_test = self.test_data[feature_cols]  # 保证是DataFrame
                X_test_scaled = self.controller.scaler.transform(X_test)
                self.test_result = self.controller.umap_model.transform(X_test_scaled)
            else:
                raise ValueError("未支持的降维方法: " + str(reduction_method))
            # 关闭进度条窗口
            progress_window.destroy()
            # 更新图表
            self.update_plot()
        except Exception as e:
            # 如果出现错误，显示错误消息
            tk.messagebox.showerror("Error", str(e))
            if 'progress_window' in locals():
                progress_window.destroy()
            return
    
    def on_visibility(self, event):
        """当页面变为可见时更新图表"""
        if event.state == '8' and self.controller.data is not None:  # 8 表示可见
            if self.fig is None:
                self.update_plot()

    def on_mouse_move(self, event):
        """处理鼠标悬停事件"""
        if event.inaxes == self.ax:
            if self.annotation:
                self.annotation.remove()
                self.annotation = None
                self.canvas.draw_idle()
            
            if not hasattr(self, 'test_scatter'):  # 修改这里，检查 test_scatter 而不是 sc
                return
            
            cont, ind = self.test_scatter.contains(event)  # 修改这里，使用 test_scatter
            
            if cont:
                idx = ind["ind"][0]
                
                # 获取点的数据
                point_data = self.test_data.iloc[idx]
                
                # 修改显示信息中的标签
                info_text = f"Distance: {self.distances[idx]:.4f}\n"
                info_text += f"Predicted MAE: {self.predicted_errors[idx]:.4f}\n"  # 修改这里
                
                # 获取前5个特征值
                feature_cols = [col for col in self.test_data.columns 
                              if col not in ['MAE', 'MSE']]
                top_features = "\n".join([f"{col}: {point_data[col]:.4f}" 
                                        for col in feature_cols[:5]])
                
                # 如果特征数量超过5个，添加省略号
                if len(feature_cols) > 5:
                    info_text += f"\n{top_features}\n..."
                else:
                    info_text += f"\n{top_features}"
                
                # 获取点的坐标
                x, y = self.test_result[idx, 0], self.test_result[idx, 1]  # 使用降维后的坐标
                
                # 计算注释框的最佳位置
                point_pos = self.ax.transData.transform((x, y))
                fig_pos = self.ax.get_window_extent()
                
                # 默认偏移量
                x_offset = 10
                y_offset = 10
                
                # 如果点在右半部分，将注释框放在左边
                if point_pos[0] > fig_pos.x0 + fig_pos.width * 0.5:
                    x_offset = -100
                
                # 如果点在上半部分，将注释框放在下边
                if point_pos[1] > fig_pos.y0 + fig_pos.height * 0.5:
                    y_offset = -100
                
                # 创建注释框
                self.annotation = self.ax.annotate(info_text,
                    xy=(x, y),
                    xytext=(x_offset, y_offset), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", 
                             fc="yellow", 
                             alpha=0.7),
                    arrowprops=dict(arrowstyle="->", 
                                  connectionstyle="arc3,rad=0",
                                  alpha=0.7),
                    zorder=1000)
                
                # 使用blit加速重绘
                if not hasattr(self, "background"):
                    self.background = self.canvas.copy_from_bbox(self.ax.bbox)
                
                self.canvas.restore_region(self.background)
                self.annotation.draw(self.canvas.get_renderer())
                self.canvas.blit(self.ax.bbox)
            
            elif self.controller.center_point is not None:
                center_x, center_y = self.controller.center_point
                distance = np.sqrt((event.xdata - center_x)**2 + (event.ydata - center_y)**2)
                if distance < 0.5:
                    info_text = "Center Point\n" + \
                               f"t-SNE-1: {center_x:.4f}\n" + \
                               f"t-SNE-2: {center_y:.4f}"
                    
                    self.annotation = self.ax.annotate(info_text,
                        xy=(center_x, center_y),
                        xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", 
                                 fc="yellow", 
                                 alpha=0.7),
                        arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0"),
                        zorder=1000)
            
            self.canvas.draw_idle()

    def toggle_train_points(self):
        """切换训练集点的显示状态"""
        if hasattr(self, 'train_scatter'):
            self.show_train.set(not self.show_train.get())
            if self.show_train.get():
                self.toggle_button.configure(text="Hide Training Set")
                self.train_scatter.set_visible(True)
                # 更新图例，包含所有元素
                self.ax.legend(loc='upper right', 
                             bbox_to_anchor=(0.98, 0.98),
                             framealpha=0.8,
                             title="Sample Types")
            else:
                self.toggle_button.configure(text="Show Training Set")
                self.train_scatter.set_visible(False)
                # 更新图例，只包含可见元素
                handles, labels = self.ax.get_legend_handles_labels()
                visible_handles = [h for h, l in zip(handles, labels) 
                                 if l != 'Training Set']
                visible_labels = [l for l in labels if l != 'Training Set']
                self.ax.legend(visible_handles, visible_labels,
                             loc='upper right',
                             bbox_to_anchor=(0.98, 0.98),
                             framealpha=0.8,
                             title="Sample Types")
            self.canvas.draw()

    def predict_errors(self, distances):
        """使用Page2的拟合曲线预测误差"""
        # 获取Page2的拟合类型和参数
        page2 = self.controller.pages["page2"]
        fit_type = page2.fit_type.get()
        regression_method = page2.regression_method.get()
        
        # 获取训练数据
        train_distances = page2.calculate_distances()
        train_errors = self.controller.data['MAE'].values
        
        # 根据拟合类型准备数据
        try:
            if fit_type == "线性":
                X = train_distances.reshape(-1, 1)
                X_pred = distances.reshape(-1, 1)
                y_train = train_errors
            elif fit_type == "多项式":
                X = np.column_stack([train_distances, train_distances**2])
                X_pred = np.column_stack([distances, distances**2])
                y_train = train_errors
            elif fit_type == "指数":
                X = train_distances.reshape(-1, 1)
                X_pred = distances.reshape(-1, 1)
                # 确保误差值为正
                min_error = np.min(train_errors[train_errors > 0])
                y_train = np.log(train_errors + min_error * 0.1)
            else:  # 对数
                # 确保距离值为正
                min_distance = np.min(train_distances[train_distances > 0])
                valid_mask = train_distances > 0
                
                # 对于训练数据
                X = np.log(train_distances[valid_mask].reshape(-1, 1))
                y_train = train_errors[valid_mask]
                
                # 对于预测数据
                X_pred = np.log(np.maximum(distances, min_distance * 0.1).reshape(-1, 1))
            
            # 选择回归方法
            if regression_method == "最小二乘":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif regression_method == "RANSAC":
                from sklearn.linear_model import RANSACRegressor
                model = RANSACRegressor(random_state=42)
            elif regression_method == "Huber":
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor(epsilon=1.35)
            else:  # Theil-Sen
                from sklearn.linear_model import TheilSenRegressor
                model = TheilSenRegressor(random_state=42)
            
            # 拟合模型并预测
            if fit_type == "指数":
                model.fit(X, y_train)
                predicted_errors = np.exp(model.predict(X_pred)) - min_error * 0.1
            else:
                model.fit(X, y_train)
                predicted_errors = model.predict(X_pred)
            
            # 确保预测值非负
            predicted_errors = np.maximum(predicted_errors, 0)
            
            return predicted_errors
            
        except Exception as e:
            tk.messagebox.showerror("错误", f"预测误差时发生错误: {str(e)}")
            return np.zeros_like(distances)

    def export_images(self, filename="test_data_analysis.png"):
        """导出当前图像为高分辨率图片"""
        try:
            if self.fig is not None:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                tk.messagebox.showinfo("Export Success", f"Image successfully exported as {filename}")
            else:
                tk.messagebox.showerror("Export Error", "No figure to export.")
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Error exporting image: {str(e)}")

class Page4(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # 创建控制面板
        self.control_frame = ttk.LabelFrame(self, text="Split‑Violin Diagnosis Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        
        # 只需要一个运行按钮
        ttk.Button(self.control_frame, text="Run Split‑Violin Diagnosis", 
                  command=self.run_analysis).grid(row=0, column=0, pady=10)
        
        # 创建带滚动条的画布容器
        self.canvas_container = ttk.Frame(self)
        self.canvas_container.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 创建垂直滚动条
        self.v_scrollbar = ttk.Scrollbar(self.canvas_container, orient="vertical")
        self.v_scrollbar.pack(side="right", fill="y")
        
        # 创建画布
        self.canvas = tk.Canvas(self.canvas_container)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # 配置滚动条
        self.v_scrollbar.config(command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        # 创建内容框架
        self.plot_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.plot_frame, 
                                                     anchor="nw")
        
        # 配置网格权重
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        # 初始化变量
        self.figs = []
        self.canvases = []
        
        # 绑定事件
        self.plot_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.bind_mouse_wheel()
    
    def on_frame_configure(self, event=None):
        """配置画布滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """当画布大小改变时，调整内容框架的宽度"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def bind_mouse_wheel(self):
        """绑定鼠标滚轮事件"""
        def on_mouse_wheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # 绑定鼠标滚轮事件到画布和所有子部件
        self.canvas.bind_all("<MouseWheel>", on_mouse_wheel)

    def add_plots_to_frame(self, results):
        try:
            # 单图半小提琴合并：results 元素为 (fig, wasserstein_val, feature)
            wasserstein_list = []
            for fig, wdist, feature in results:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, pady=4)
                self.figs.append(fig)
                self.canvases.append(canvas)
                if wdist is not None and np.isfinite(wdist):
                    wasserstein_list.append(wdist)
            if wasserstein_list:
                avg_distance = np.mean(wasserstein_list)
                avg_label = ttk.Label(self.plot_frame, text=f"平均Wasserstein距离: {avg_distance:.4f}", font=("Arial", 12, "bold"), foreground="red")
                avg_label.pack(side="top", pady=8)
        except Exception as e:
            raise Exception(f"Failed to add plots to frame: {str(e)}")
    
    def run_analysis(self):
        """运行分析"""
        if self.controller.data is None:
            tk.messagebox.showerror("错误", "请先在页面1加载训练集")
            return
            
        try:
            # 创建进度条窗口
            progress_window = tk.Toplevel(self)
            progress_window.title("分析进度")
            progress_window.geometry("300x150")
            progress_window.transient(self)
            progress_window.grab_set()
            
            # 禁用关闭按钮
            progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
            
            # 居中显示
            progress_window.geometry("+%d+%d" % (
                self.winfo_rootx() + 50,
                self.winfo_rooty() + 50))
            
            # 添加进度条和标签
            label = ttk.Label(progress_window, text="正在准备分析...")
            label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(
                progress_window, 
                length=200, 
                mode='indeterminate'
            )
            progress_bar.pack(pady=10)
            
            # 开始进度条动画
            progress_bar.start(10)
            progress_window.update()
                    
            def compute_analysis():
                try:
                    # 清除现有图表
                    for fig in self.figs:
                        plt.close(fig)
                    self.figs.clear()
                    self.canvases.clear()
                    self.after_idle(lambda: clear_plot_frame())

                    # 获取特征列
                    feature_cols = [col for col in self.controller.data.columns 
                                  if col not in ['MAE', 'MSE', 'y']]
                    results = []  # 存储所有图表结果

                    # 获取Page3的测试数据
                    page3 = self.controller.pages.get("page3")
                    test_data = page3.test_data if page3 else None

                    # 创建每个特征的分布图
                    for i, feature in enumerate(feature_cols):
                        self.after_idle(lambda i=i, f=feature: label.config(
                            text=f"Analyzing feature: {f} ({i+1}/{len(feature_cols)})"))
                        self.after_idle(lambda: progress_window.update())
                        result = create_feature_plot(feature, test_data)
                        results.append(result)
                    # 修改这里：使用self.add_plots_to_frame而不是直接调用add_plots_to_frame
                    self.after_idle(lambda: self.add_plots_to_frame(results))
                    self.after_idle(complete_analysis)
                except Exception as e:
                            self.after_idle(lambda: handle_error(str(e)))
                    
            def clear_plot_frame():
                for widget in self.plot_frame.winfo_children():
                    widget.destroy()

            def create_feature_plot(feature, test_data=None):
                try:
                    from scipy.stats import wasserstein_distance, gaussian_kde
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # 为右侧 Wasserstein 文本预留空间
                    try:
                        fig.subplots_adjust(right=0.80)
                    except Exception:
                        pass
                    # 固定子图纵横比，避免窗口变宽时图像过度拉伸
                    try:
                        ax.set_box_aspect(0.75)  # height/width = 3/4
                    except Exception:
                        # 旧版本 Matplotlib 兼容：等比例作为兜底
                        ax.set_aspect('equal', adjustable='box')

                    train_series = self.controller.data[feature].dropna()
                    test_series = None
                    if test_data is not None and feature in test_data.columns:
                        test_series = test_data[feature].dropna()

                    if len(train_series) + (0 if test_series is None else len(test_series)) < 2:
                        ax.text(0.5, 0.5, "数据不足", ha='center', va='center', color='red', transform=ax.transAxes)
                        return fig, None, feature

                    # y 轴范围
                    if test_series is not None and len(test_series) > 0:
                        all_data = pd.concat([train_series, test_series])
                    else:
                        all_data = train_series
                    y_min, y_max = all_data.min(), all_data.max()
                    y = np.linspace(y_min, y_max, 200)
                    violin_width = 0.2
                    scatter_max_offset = 0.125

                    # 左半边（训练集）
                    if len(train_series) > 1:
                        kde_train = gaussian_kde(train_series)
                        v_train = kde_train(y)
                        v_train = v_train / v_train.max() * violin_width
                        ax.fill_betweenx(y, 1 - v_train, 1, facecolor='lightgreen', alpha=0.7)
                        # 半箱线（左）
                        q1, med, q3 = np.percentile(train_series, [25, 50, 75])
                        box_w = violin_width * 0.2
                        ax.plot([1 - box_w, 1], [q1, q1], color='black', lw=1)
                        ax.plot([1 - box_w, 1], [q3, q3], color='black', lw=1)
                        ax.plot([1 - box_w, 1 - box_w], [q1, q3], color='black', lw=1)
                        ax.plot([1, 1], [q1, q3], color='black', lw=1)
                        ax.plot([1 - box_w, 1], [med, med], color='orange', lw=1)

                    # 右半边（测试集）
                    wasserstein_val = None
                    if test_series is not None and len(test_series) > 1:
                        kde_test = gaussian_kde(test_series)
                        v_test = kde_test(y)
                        v_test = v_test / v_test.max() * violin_width
                        ax.fill_betweenx(y, 1, 1 + v_test, facecolor='lightblue', alpha=0.7)
                        # 半箱线（右）
                        q1, med, q3 = np.percentile(test_series, [25, 50, 75])
                        box_w = violin_width * 0.2
                        ax.plot([1, 1 + box_w], [q1, q1], color='black', lw=1)
                        ax.plot([1, 1 + box_w], [q3, q3], color='black', lw=1)
                        ax.plot([1 + box_w, 1 + box_w], [q1, q3], color='black', lw=1)
                        ax.plot([1, 1], [q1, q3], color='black', lw=1)
                        ax.plot([1, 1 + box_w], [med, med], color='orange', lw=1)

                        wasserstein_val = wasserstein_distance(train_series, test_series)
                        # 将 Wasserstein 距离放置在图像右侧，避免与图例重叠
                        try:
                            fig.text(0.825, 0.5, f"Wasserstein\n{wasserstein_val:.4f}", va='center', ha='left', fontsize=11, color='red')
                        except Exception:
                            pass
                    elif test_series is None or len(test_series) == 0:
                        ax.text(1.18, (y_min + y_max) / 2, "No test data", ha='center', va='center', color='red')

                    # 点云分布（分行抽样）
                    n_rows = 15
                    max_per_row = 20
                    left_color = 'green'
                    right_color = 'blue'
                    point_size = 15
                    alpha = 0.3

                    row_edges = np.linspace(y_min, y_max, n_rows + 1)
                    for i in range(n_rows):
                        y_low, y_high = row_edges[i], row_edges[i + 1]
                        left_points = train_series[(train_series >= y_low) & (train_series < y_high)]
                        right_points = (test_series[(test_series >= y_low) & (test_series < y_high)]
                                        if test_series is not None else pd.Series([], dtype=float))

                        n_left, n_right = len(left_points), len(right_points)
                        total = n_left + n_right
                        if total == 0:
                            continue
                        if n_left > 0 and n_right > 0:
                            show_left = max(1, min(int(np.round(n_left / total * max_per_row)), n_left))
                            show_right = max(1, min(max_per_row - show_left, n_right))
                        elif n_left > 0:
                            show_left, show_right = min(n_left, max_per_row), 0
                        else:
                            show_left, show_right = 0, min(n_right, max_per_row)

                        if show_left > 0:
                            y_vals = (np.random.choice(left_points, show_left, replace=False)
                                      if n_left > show_left else left_points.values)
                            jitter = np.abs(np.random.normal(0, scatter_max_offset/4, size=show_left))
                            jitter = np.clip(jitter, 0, scatter_max_offset)
                            x_vals = 1 - jitter
                            ax.scatter(x_vals, y_vals, alpha=alpha, color=left_color, s=point_size, label='Train' if i == 0 else "")
                        if show_right > 0:
                            y_vals = (np.random.choice(right_points, show_right, replace=False)
                                      if n_right > show_right else right_points.values)
                            jitter = np.abs(np.random.normal(0, scatter_max_offset/4, size=show_right))
                            jitter = np.clip(jitter, 0, scatter_max_offset)
                            x_vals = 1 + jitter
                            ax.scatter(x_vals, y_vals, alpha=alpha, color=right_color, s=point_size, label='Predict' if i == 0 else "")

                    # 轴样式与标题
                    ax.set_xlim(1 - 0.25, 1 + 0.25)
                    x_ticks = np.linspace(1 - 0.25, 1 + 0.25, 5)
                    ax.set_xticks(x_ticks)
                    if y_max > y_min:
                        y_ticks = np.linspace(y_min, y_max, 6)
                        ax.set_yticks(y_ticks)
                    ax.set_xlabel('Demension1')
                    ax.set_ylabel('Demension2')
                    ax.set_title(f"{feature} Distribution Comparison")
                    ax.grid(False)

                    import matplotlib.patches as mpatches
                    handles = [
                        mpatches.Patch(color='lightgreen', label='Train'),
                        mpatches.Patch(color='lightblue', label='Predict')
                    ]
                    ax.legend(handles=handles)

                    fig.tight_layout()
                    return fig, wasserstein_val, feature
                except Exception as e:
                    raise Exception(f"Failed to create feature plot ({feature}): {str(e)}")
            
            def complete_analysis():
                self.plot_frame.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                progress_window.destroy()
            
            def handle_error(error_msg):
                progress_window.destroy()
                tk.messagebox.showerror("错误", f"分析失败: {error_msg}")
            
            thread = threading.Thread(target=compute_analysis)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            tk.messagebox.showerror("错误", f"分析过程中发生错误: {str(e)}")
    
    def on_mouse_move(self, event):
        """处理鼠标悬停事件"""
        if event.inaxes == self.ax:
            if self.annotation:
                self.annotation.remove()
                self.annotation = None
                self.canvas.draw_idle()
            
            if not hasattr(self, 'test_scatter'):  # 修改这里，检查 test_scatter 而不是 sc
                return
            
            cont, ind = self.test_scatter.contains(event)  # 修改这里，使用 test_scatter
            
            if cont:
                idx = ind["ind"][0]
                
                # 获取点的数据
                point_data = self.test_data.iloc[idx]
                
                # 修改显示信息中的标签
                info_text = f"Distance: {self.distances[idx]:.4f}\n"
                info_text += f"Predicted MAE: {self.predicted_errors[idx]:.4f}\n"  # 修改这里
                
                # 获取前5个特征值
                feature_cols = [col for col in self.test_data.columns 
                              if col not in ['MAE', 'MSE']]
                top_features = "\n".join([f"{col}: {point_data[col]:.4f}" 
                                        for col in feature_cols[:5]])
                
                # 如果特征数量超过5个，添加省略号
                if len(feature_cols) > 5:
                    info_text += f"\n{top_features}\n..."
                else:
                    info_text += f"\n{top_features}"
                
                # 获取点的坐标
                x, y = self.test_result[idx, 0], self.test_result[idx, 1]  # 使用降维后的坐标
                
                # 计算注释框的最佳位置
                point_pos = self.ax.transData.transform((x, y))
                fig_pos = self.ax.get_window_extent()
                
                # 默认偏移量
                x_offset = 10
                y_offset = 10
                
                # 如果点在右半部分，将注释框放在左边
                if point_pos[0] > fig_pos.x0 + fig_pos.width * 0.5:
                    x_offset = -100
                
                # 如果点在上半部分，将注释框放在下边
                if point_pos[1] > fig_pos.y0 + fig_pos.height * 0.5:
                    y_offset = -100
                
                # 创建注释框
                self.annotation = self.ax.annotate(info_text,
                    xy=(x, y),
                    xytext=(x_offset, y_offset), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", 
                             fc="yellow", 
                             alpha=0.7),
                    arrowprops=dict(arrowstyle="->", 
                                  connectionstyle="arc3,rad=0",
                                  alpha=0.7),
                    zorder=1000)
                
                # 使用blit加速重绘
                if not hasattr(self, "background"):
                    self.background = self.canvas.copy_from_bbox(self.ax.bbox)
                
                self.canvas.restore_region(self.background)
                self.annotation.draw(self.canvas.get_renderer())
                self.canvas.blit(self.ax.bbox)
            
            elif self.controller.center_point is not None:
                center_x, center_y = self.controller.center_point
                distance = np.sqrt((event.xdata - center_x)**2 + (event.ydata - center_y)**2)
                if distance < 0.5:
                    info_text = "Center Point\n" + \
                               f"t-SNE-1: {center_x:.4f}\n" + \
                               f"t-SNE-2: {center_y:.4f}"
                    
                    self.annotation = self.ax.annotate(info_text,
                        xy=(center_x, center_y),
                        xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", 
                                 fc="yellow", 
                                 alpha=0.7),
                        arrowprops=dict(arrowstyle="->", 
                                      connectionstyle="arc3,rad=0"),
                        zorder=1000)
            
            self.canvas.draw_idle()

    def toggle_train_points(self):
        """切换训练集点的显示状态"""
        if hasattr(self, 'train_scatter'):
            self.show_train.set(not self.show_train.get())
            if self.show_train.get():
                self.toggle_button.configure(text="Hide Training Set")
                self.train_scatter.set_visible(True)
                # 更新图例，包含所有元素
                self.ax.legend(loc='upper right', 
                             bbox_to_anchor=(0.98, 0.98),
                             framealpha=0.8,
                             title="Sample Types")
            else:
                self.toggle_button.configure(text="Show Training Set")
                self.train_scatter.set_visible(False)
                # 更新图例，只包含可见元素
                handles, labels = self.ax.get_legend_handles_labels()
                visible_handles = [h for h, l in zip(handles, labels) 
                                 if l != 'Training Set']
                visible_labels = [l for l in labels if l != 'Training Set']
                self.ax.legend(visible_handles, visible_labels,
                             loc='upper right',
                             bbox_to_anchor=(0.98, 0.98),
                             framealpha=0.8,
                             title="Sample Types")
            self.canvas.draw()

    def predict_errors(self, distances):
        """使用Page2的拟合曲线预测误差"""
        # 获取Page2的拟合类型和参数
        page2 = self.controller.pages["page2"]
        fit_type = page2.fit_type.get()
        regression_method = page2.regression_method.get()
        
        # 获取训练数据
        train_distances = page2.calculate_distances()
        train_errors = self.controller.data['MAE'].values
        
        # 根据拟合类型准备数据
        try:
            if fit_type == "线性":
                X = train_distances.reshape(-1, 1)
                X_pred = distances.reshape(-1, 1)
                y_train = train_errors
            elif fit_type == "多项式":
                X = np.column_stack([train_distances, train_distances**2])
                X_pred = np.column_stack([distances, distances**2])
                y_train = train_errors
            elif fit_type == "指数":
                X = train_distances.reshape(-1, 1)
                X_pred = distances.reshape(-1, 1)
                # 确保误差值为正
                min_error = np.min(train_errors[train_errors > 0])
                y_train = np.log(train_errors + min_error * 0.1)
            else:  # 对数
                # 确保距离值为正
                min_distance = np.min(train_distances[train_distances > 0])
                valid_mask = train_distances > 0
                
                # 对于训练数据
                X = np.log(train_distances[valid_mask].reshape(-1, 1))
                y_train = train_errors[valid_mask]
                
                # 对于预测数据
                X_pred = np.log(np.maximum(distances, min_distance * 0.1).reshape(-1, 1))
            
            # 选择回归方法
            if regression_method == "最小二乘":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif regression_method == "RANSAC":
                from sklearn.linear_model import RANSACRegressor
                model = RANSACRegressor(random_state=42)
            elif regression_method == "Huber":
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor(epsilon=1.35)
            else:  # Theil-Sen
                from sklearn.linear_model import TheilSenRegressor
                model = TheilSenRegressor(random_state=42)
            
            # 拟合模型并预测
            if fit_type == "指数":
                model.fit(X, y_train)
                predicted_errors = np.exp(model.predict(X_pred)) - min_error * 0.1
            else:
                model.fit(X, y_train)
                predicted_errors = model.predict(X_pred)
            
            # 确保预测值非负
            predicted_errors = np.maximum(predicted_errors, 0)
            
            return predicted_errors
            
        except Exception as e:
            tk.messagebox.showerror("错误", f"预测误差时发生错误: {str(e)}")
            return np.zeros_like(distances)

class Page5(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # 创建控制面板
        self.control_frame = ttk.LabelFrame(self, text="Exploration‑based Sampling Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        
        # 设置统一的按钮宽度
        btn_width = 20
        
        # 添加输入框和标签
        input_frame = ttk.Frame(self.control_frame)
        input_frame.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Number of Points:").grid(row=0, column=0, padx=5, pady=5)
        self.num_points_var = tk.StringVar(value="1")
        ttk.Entry(input_frame, textvariable=self.num_points_var, width=5).grid(row=0, column=1, padx=5, pady=5)
        
        # 添加按钮，统一宽度和间距
        self.manual_btn = ttk.Button(self.control_frame, 
                                   text="Manual Completion",
                                   width=btn_width,
                                   command=self.toggle_manual_mode)
        self.manual_btn.grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(self.control_frame, 
                  text="Auto Complete Points",
                  width=btn_width,
                  command=self.run_active_sampling).grid(row=2, column=0, padx=5, pady=5)
        
        ttk.Button(self.control_frame, 
                  text="Export Completed Points",
                  width=btn_width,
                  command=self.export_new_point).grid(row=3, column=0, padx=5, pady=5)
        
        # 创建图表区域
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.stats_frame = ttk.LabelFrame(self, text="Supplementary Points (Exploration)", padding="5")
        self.stats_frame.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.canvas = None
        self.fig = None
        self.ax = None
        self.new_points_hd = []
        self.new_points_2d = []
        self.new_points_type = []  # 新增：记录每个补全点类型（"auto"/"manual"）
        self.triangle_indices = None
        self.manual_triangle = []
        self.manual_mode = False
        self.highlight_idx = None
        from tkinter import StringVar
        self.tree = ttk.Treeview(self.stats_frame, columns=("idx", "x", "y", "mean", "type"), show="headings", height=15)
        self.tree.heading("idx", text="ID")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("mean", text="Feature Mean")
        self.tree.heading("type", text="Type")
        self.tree.column("idx", width=40, anchor="center")
        self.tree.column("x", width=70, anchor="center")
        self.tree.column("y", width=70, anchor="center")
        self.tree.column("mean", width=90, anchor="center")
        self.tree.column("type", width=60, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Delete Point", command=self.delete_selected_point)
        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.create_plot_area()
        self.bind("<Visibility>", self.on_visibility)
        self.tsne_result_raw = None  # 新增：保存原始点2D坐标
        self.tsne_result_new = []    # 新增：保存补全点2D坐标
        self.all_points_2d = []  # 新增：所有点2D坐标
        self.all_points_type = [] # 新增：所有点类型
    def toggle_manual_mode(self):
        if not self.manual_mode:
            self.manual_mode = True
            self.manual_triangle = []
            self.manual_btn.config(text="Finish Manual Completion")
            tk.messagebox.showinfo("Info", "Manual completion mode: Please click 3 vertices in the plot (can be original or completed points), a new point will be added for each 3 points. Click 'Finish Manual Completion' to exit manual mode.")
        else:
            self.manual_mode = False
            self.manual_triangle = []
            self.manual_btn.config(text="Manual Completion")

    def create_plot_area(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
    def on_click(self, event):
        if not self.manual_mode:
            return
        if event.inaxes != self.ax:
            return
        # 支持选择原始点和所有补全点
        tsne_result = self.controller.tsne_result
        all_points_2d = list(tsne_result) + self.new_points_2d
        all_points_type = ["raw"] * len(tsne_result) + self.new_points_type
        dists = [np.linalg.norm(np.array([event.xdata, event.ydata]) - np.array(pt2d)) if pt2d is not None else 1e10 for pt2d in all_points_2d]
        idx = int(np.argmin(dists))
        # 记录为全局索引（前面是原始点，后面是补全点）
        self.manual_triangle.append(idx)
        if len(self.manual_triangle) == 3:
            self.triangle_indices = tuple(self.manual_triangle)
            self.manual_triangle = []
            self.run_active_sampling(manual=True)
    def run_active_sampling(self, manual=False):
        data = self.controller.data
        tsne_result = self.controller.tsne_result
        if data is None or tsne_result is None:
            tk.messagebox.showerror("Error", "Please finish dimensionality reduction analysis on Page 1 first.")
            return
        X = data[[col for col in data.columns if col not in ['MAE', 'MSE']]].values
        n_raw = len(X)
        try:
            num_points = int(self.num_points_var.get())
        except Exception:
            num_points = 1
        triangles = []
        tsne_for_triangulation = tsne_result[:n_raw]
        if manual and self.triangle_indices is not None:
            triangles = [self.triangle_indices]
        else:
            from scipy.spatial import Delaunay
            tri = Delaunay(tsne_for_triangulation)
            def triangle_area(pts):
                a, b, c = pts
                return 0.5 * np.abs((a[0]-c[0])*(b[1]-a[1]) - (a[0]-b[0])*(c[1]-a[1]))
            areas = np.array([triangle_area(tsne_for_triangulation[simplex]) for simplex in tri.simplices])
            used = []
            for i in np.argsort(areas)[::-1]:
                tri_idx = tri.simplices[i]
                new_hd = X[tri_idx].mean(axis=0)
                if any(np.allclose(new_hd, hd) for hd in self.new_points_hd):
                    continue
                used.append(tri_idx)
                if len(used) == num_points:
                    break
            triangles = used
        method = self.controller.pages["page1"].dim_reduction_var.get()
        new_hd_list = []
        for tri_idx in triangles:
            if manual:
                hd_list = []
                for i in tri_idx:
                    if i < n_raw:
                        hd = X[i]
                    else:
                        hd = self.new_points_hd[i - n_raw]
                    hd_list.append(hd)
                new_hd = np.mean(hd_list, axis=0)
            else:
                new_hd = X[tri_idx].mean(axis=0)
            if any(np.allclose(new_hd, hd) for hd in self.new_points_hd):
                continue
            new_hd_list.append(new_hd)
            self.new_points_hd.append(new_hd)
            self.new_points_type.append("auto" if not manual else "manual")
        # t-SNE整体重降维，分开保存原始点和补全点2D坐标
        if method == "t-SNE" and len(new_hd_list) > 0:
            from sklearn.manifold import TSNE
            X_aug = np.vstack([X, self.new_points_hd])
            perplexity = float(self.controller.pages["page1"].perplexity_var.get())
            tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=perplexity)
            tsne_result_aug = tsne.fit_transform(X_aug)
            self.tsne_result_raw = tsne_result_aug[:n_raw]  # 原始点
            # 只保留当前所有补全点的2D坐标，不再累加
            self.tsne_result_new = [tsne_result_aug[n_raw + i] for i in range(len(self.new_points_hd))]
            self.controller.tsne_result = self.tsne_result_raw
            self.new_points_2d = self.tsne_result_new.copy()
            # 统一维护所有点和类型
            self.all_points_2d = list(self.tsne_result_raw) + list(self.tsne_result_new)
            self.all_points_type = ["Original"]*n_raw + ["Completed"]*len(self.tsne_result_new)
        elif method in ("PCA", "UMAP"):
            self.all_points_2d = list(self.controller.tsne_result)
            if self.new_points_hd:
                new_scaled = self.controller.scaler.transform(self.new_points_hd)
                if method == "PCA":
                    new_2d = self.controller.pca_model.transform(new_scaled)
                else:
                    new_2d = self.controller.umap_model.transform(new_scaled)
                self.new_points_2d = list(new_2d)
                self.all_points_2d += self.new_points_2d
            self.all_points_type = ["Original"]*n_raw + ["Completed"]*len(self.new_points_hd)
        self.update_stats()
        self.triangle_indices = None
        self.update_plot()
        # 主动通知Page6刷新
        page6 = self.controller.pages.get("page6")
        if page6 is not None:
            page6.refresh_all_points()
            page6.update_plot()
    def update_stats(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, (hd, pt2d, typ) in enumerate(zip(self.new_points_hd, self.new_points_2d, self.new_points_type)):
            mean_val = np.mean(hd)
            x, y = pt2d if pt2d is not None else ("-", "-")
            self.tree.insert("", "end", iid=str(idx), values=(idx+1, f"{x:.3f}" if x!="-" else x, f"{y:.3f}" if y!="-" else y, f"{mean_val:.3f}", "Manual" if typ=="manual" else "Auto"))
    def delete_selected_point(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        del self.new_points_hd[idx]
        del self.new_points_2d[idx]
        del self.new_points_type[idx]
        self.highlight_idx = None
        self.update_stats()
        self.update_plot()
    def update_plot(self):
        self.ax.clear()
        method = self.controller.pages["page1"].dim_reduction_var.get()
        tsne_result = self.controller.tsne_result
        X = self.controller.data[[col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]].values
        if method == "t-SNE":
            # t-SNE时，所有点（原始+补全）都显示，补全点高亮
            n_raw = len(X)
            # 画原始点
            self.ax.scatter(tsne_result[:n_raw,0], tsne_result[:n_raw,1], alpha=0.3, label="Original Points")
            # 画补全点
            for idx, (new_2d, typ) in enumerate(zip(self.new_points_2d, self.new_points_type)):
                if new_2d is not None:
                    if self.highlight_idx == idx:
                        self.ax.scatter(new_2d[0], new_2d[1], color='yellow', marker='*', s=350, label='Highlighted Point', edgecolors='red', linewidths=2, zorder=10)
                    elif typ == "manual":
                        self.ax.scatter(new_2d[0], new_2d[1], color='blue', marker='*', s=200, label='Manual Point' if idx==0 else "_nolegend_", zorder=5)
                    else:
                        self.ax.scatter(new_2d[0], new_2d[1], color='red', marker='*', s=200, label='Auto Point' if idx==0 else "_nolegend_", zorder=5)
                else:
                    self.ax.text(0.05, 0.95, "Cannot display 2D position", transform=self.ax.transAxes, color='red')
        else:
            # PCA/UMAP时，原有逻辑
            tsne_result = self.controller.tsne_result
            self.ax.scatter(tsne_result[:,0], tsne_result[:,1], alpha=0.3, label="Original Points")
            if self.triangle_indices is not None:
                tri_pts = tsne_result[self.triangle_indices]
                self.ax.plot(*np.vstack((tri_pts, tri_pts[0])).T, color='orange', lw=2, label='Selected Triangle')
            for idx, (new_2d, typ) in enumerate(zip(self.new_points_2d, self.new_points_type)):
                if new_2d is not None:
                    if self.highlight_idx == idx:
                        self.ax.scatter(new_2d[0], new_2d[1], color='yellow', marker='*', s=350, label='Highlighted Point', edgecolors='red', linewidths=2, zorder=10)
                    elif typ == "manual":
                        self.ax.scatter(new_2d[0], new_2d[1], color='blue', marker='*', s=200, label='Manual Point' if idx==0 else "_nolegend_", zorder=5)
                    else:
                        self.ax.scatter(new_2d[0], new_2d[1], color='red', marker='*', s=200, label='Auto Point' if idx==0 else "_nolegend_", zorder=5)
                else:
                    self.ax.text(0.05, 0.95, "Cannot display 2D position", transform=self.ax.transAxes, color='red')
        self.ax.legend()
        self.ax.set_title("Exploration based Sampling: Data Void Completion")
        self.canvas.draw()
    def export_new_point(self):
        if not self.new_points_hd:
            tk.messagebox.showerror("Error", "Please detect and complete holes first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")])
        if file_path:
            df = pd.DataFrame(self.new_points_hd, columns=[col for col in self.controller.data.columns if col not in ['MAE', 'MSE']])
            df.to_csv(file_path, index=False)
            tk.messagebox.showinfo("Export Success", f"Completed points exported to: {file_path}")
    def show_context_menu(self, event):
        rowid = self.tree.identify_row(event.y)
        if rowid:
            self.tree.selection_set(rowid)
            self.menu.post(event.x_root, event.y_root)
    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            self.highlight_idx = None
        else:
            self.highlight_idx = int(sel[0])
        self.update_plot()
    def on_visibility(self, event):
        # 每次都同步all_points_2d/all_points_type为最新状态
        method = self.controller.pages["page1"].dim_reduction_var.get()
        if method == "t-SNE" and self.tsne_result_raw is not None:
            n_raw = len(self.tsne_result_raw)
            self.all_points_2d = list(self.tsne_result_raw) + list(self.tsne_result_new)
            self.all_points_type = ["Original"]*n_raw + ["Completed"]*len(self.tsne_result_new)
        elif method in ("PCA", "UMAP") and self.controller.tsne_result is not None:
            n_raw = len(self.controller.tsne_result)
            self.all_points_2d = list(self.controller.tsne_result)
            if self.new_points_hd:
                new_scaled = self.controller.scaler.transform(self.new_points_hd)
                if method == "PCA":
                    new_2d = self.controller.pca_model.transform(new_scaled)
                else:
                    new_2d = self.controller.umap_model.transform(new_scaled)
                self.new_points_2d = list(new_2d)
                self.all_points_2d += self.new_points_2d
            self.all_points_type = ["Original"]*n_raw + ["Completed"]*len(self.new_points_hd)
        self.update_plot()

class Page6(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # 创建控制面板
        self.control_frame = ttk.LabelFrame(self, text="Redundancy reduced Sampling Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        
        # 设置统一的按钮宽度
        btn_width = 20
        
        # 添加输入框和标签
        input_frame = ttk.Frame(self.control_frame)
        input_frame.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Number of Samples:").grid(row=0, column=0, padx=5, pady=5)
        self.n_samples_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.n_samples_var, width=6).grid(row=0, column=1, padx=5, pady=5)
        
        # 添加按钮，统一宽度和间距
        ttk.Button(self.control_frame, 
                  text="Run Redundancy reduced Sampling",
                  width=btn_width,
                  command=self.active_sampling_dialog).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(self.control_frame, 
                  text="Export Core Set",
                  width=btn_width,
                  command=self.export_sampled_points).grid(row=2, column=0, padx=5, pady=5)
        
        # 添加隐藏其他点的按钮
        self.hide_others = tk.BooleanVar(value=False)
        self.toggle_button = ttk.Button(self.control_frame,
                                      text="Hide Other Points",
                                      width=btn_width,
                                      command=self.toggle_other_points)
        self.toggle_button.grid(row=3, column=0, padx=5, pady=5)
        
        # 创建图表区域
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        self.stats_frame = ttk.LabelFrame(self, text="Selected Points (Core Set)", padding="5")
        self.stats_frame.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.canvas = None
        self.fig = None
        self.ax = None
        self.selected_indices = [] # 存储已选择点的索引（在all_points_2d中的索引）
        self.highlight_idx = None
        self.all_points_2d = []
        self.all_points_type = []
        self.tree = ttk.Treeview(self.stats_frame, columns=("idx", "x", "y", "type"), show="headings", height=15)
        self.tree.heading("idx", text="ID")
        self.tree.heading("x", text="X")
        self.tree.heading("y", text="Y")
        self.tree.heading("type", text="Type")
        self.tree.column("idx", width=40, anchor="center")
        self.tree.column("x", width=70, anchor="center")
        self.tree.column("y", width=70, anchor="center")
        self.tree.column("type", width=60, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="Delete Point", command=self.delete_selected_point)
        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.create_plot_area()
        self.bind("<Visibility>", self.on_visibility)
    def on_visibility(self, event):
        self.refresh_all_points()
        self.update_stats()
        self.update_plot()
    def create_plot_area(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def refresh_all_points(self):
        page5 = self.controller.pages.get("page5")
        self.all_points_2d = []
        self.all_points_type = []
        self.selected_indices = []
        if page5 is not None and hasattr(page5, "all_points_2d") and hasattr(page5, "all_points_type") and len(page5.all_points_2d) == len(page5.all_points_type):
            self.all_points_2d = list(page5.all_points_2d)
            self.all_points_type = list(page5.all_points_type)
            # 默认选中所有补全点
            self.selected_indices = [i for i, t in enumerate(self.all_points_type) if t == "Completed"]
        else:
            tsne_result = self.controller.tsne_result
            self.all_points_2d = list(tsne_result)
            self.all_points_type = ["Original"] * len(tsne_result)
    def active_sampling_dialog(self):
        # 采样策略选择对话框
        strategies = ["Cluster-based (KMeans)", "Diversity Sampling", "Random Sampling"]
        strategy = tkinter.simpledialog.askstring("Select Sampling Strategy", "Available strategies:\n1. Cluster-based (KMeans)\n2. Diversity Sampling\n3. Random Sampling\n\nPlease enter number or name:", initialvalue="1")
        if not strategy:
            return
        # 获取采样点个数
        try:
            n_samples = int(self.n_samples_var.get())
        except Exception:
            n_samples = 10
        if strategy.strip() in ["1", "Cluster-based (KMeans)"]:
            self.run_cluster_based_sampling(n_samples=n_samples)
        elif strategy.strip() in ["2", "Diversity Sampling"]:
            self.run_diversity_sampling(n_samples=n_samples)
        elif strategy.strip() in ["3", "Random Sampling"]:
            self.run_random_sampling(n_samples=n_samples)
        else:
            tk.messagebox.showerror("Error", "Unsupported sampling strategy")
    def run_cluster_based_sampling(self, n_samples=10):
        # KMeans聚类中心采样，只在已有点中选取，主动采样点类型标记为"主动采样"
        from sklearn.cluster import KMeans
        import numpy as np
        points = np.array([pt for pt in self.all_points_2d if pt is not None])
        if len(points) < 2:
            tk.messagebox.showerror("Error", "点数不足，无法聚类采样")
            return
        n_samples = min(n_samples, len(points))
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        centers = kmeans.cluster_centers_
        selected = []
        for i in range(n_samples):
            cluster_pts = np.where(labels == i)[0]
            if len(cluster_pts) == 0:
                continue
            dists = np.linalg.norm(points[cluster_pts] - centers[i], axis=1)
            idx = cluster_pts[np.argmin(dists)]
            selected.append(idx)
        # 保留补全点索引，新增主动采样点索引
        self.selected_indices = [i for i in self.selected_indices if self.all_points_type[i]=="Completed"] + [i for i in selected if self.all_points_type[i]!="Completed"]
        self.update_stats()
        self.update_plot()
    def run_diversity_sampling(self, n_samples=10):
        # 多样性采样，只在已有点中选取，主动采样点类型标记为"主动采样"
        import numpy as np
        points = np.array([pt for pt in self.all_points_2d if pt is not None])
        if len(points) < 2:
            tk.messagebox.showerror("Error", "点数不足，无法多样性采样")
            return
        n_samples = min(n_samples, len(points))
        selected = [0]
        for _ in range(1, n_samples):
            dists = np.min([np.linalg.norm(points - points[i], axis=1) for i in selected], axis=0)
            idx = np.argmax(dists)
            if idx in selected:
                break
            selected.append(idx)
        self.selected_indices = [i for i in self.selected_indices if self.all_points_type[i]=="Completed"] + [i for i in selected if self.all_points_type[i]!="Completed"]
        self.update_stats()
        self.update_plot()
    def run_random_sampling(self, n_samples=10):
        # 随机采样，只在已有点中选取，主动采样点类型标记为"主动采样"
        import numpy as np
        points = np.array([pt for pt in self.all_points_2d if pt is not None])
        if len(points) < 1:
            tk.messagebox.showerror("Error", "点数不足，无法随机采样")
            return
        n_samples = min(n_samples, len(points))
        selected = np.random.choice(len(points), n_samples, replace=False).tolist()
        self.selected_indices = [i for i in self.selected_indices if self.all_points_type[i]=="Completed"] + [i for i in selected if self.all_points_type[i]!="Completed"]
        self.update_stats()
        self.update_plot()
    def update_stats(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, sel_idx in enumerate(self.selected_indices):
            pt2d = self.all_points_2d[sel_idx]
            typ = self.all_points_type[sel_idx]
            x, y = pt2d if pt2d is not None else ("-", "-")
            self.tree.insert("", "end", iid=str(idx), values=(idx+1, f"{x:.3f}" if x!="-" else x, f"{y:.3f}" if y!="-" else y, typ))
    def delete_selected_point(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        del self.selected_indices[idx]
        self.highlight_idx = None
        self.update_stats()
        self.update_plot()
    def show_context_menu(self, event):
        rowid = self.tree.identify_row(event.y)
        if rowid:
            self.tree.selection_set(rowid)
            self.menu.post(event.x_root, event.y_root)
    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            self.highlight_idx = None
        else:
            self.highlight_idx = int(sel[0])
        self.update_plot()
    def update_plot(self):
        self.ax.clear()
        # 分组批量scatter，极大提升速度
        import numpy as np
        idx_map = [i for i, pt in enumerate(self.all_points_2d) if pt is not None]
        if len(idx_map) == 0:
            self.ax.set_title("Redundancy reduced Sampling: Core Set Selection")
            self.canvas.draw()
            return
        pts = np.array([self.all_points_2d[i] for i in idx_map])
        types = np.array(self.all_points_type)[idx_map]
        selected_mask = np.isin(idx_map, self.selected_indices)
        
        # Selected Points样式与Page5原始点一致
        if self.selected_indices:
            sel_pts = pts[selected_mask]
            if sel_pts.size > 0:
                self.ax.scatter(sel_pts[:,0], sel_pts[:,1], color='blue', marker='o', s=80, alpha=0.8, label='Selected Point', zorder=15)
            # 高亮
            if self.highlight_idx is not None and self.highlight_idx < len(self.selected_indices):
                sel_idx = self.selected_indices[self.highlight_idx]
                pt2d = self.all_points_2d[sel_idx]
                self.ax.scatter(pt2d[0], pt2d[1], color='yellow', marker='o', s=180, label='Highlighted', edgecolors='red', linewidths=2, zorder=20)
        
        # 如果没有隐藏其他点，显示所有点
        if not self.hide_others.get():
            # 原始点
            mask_ori = (types == "Original") & (~selected_mask)
            if np.any(mask_ori):
                self.ax.scatter(pts[mask_ori,0], pts[mask_ori,1], color='#999999', marker='o', s=80, alpha=0.3, label='Original Point', zorder=5)
            # 补全点
            mask_comp = (types == "Completed") & (~selected_mask)
            if np.any(mask_comp):
                self.ax.scatter(pts[mask_comp,0], pts[mask_comp,1], color='blue', marker='*', s=180, label='Completed Point', zorder=10)
        
        self.ax.legend()
        self.ax.set_title("Redundancy reduced Sampling: Core Set Selection")
        self.canvas.draw()
    def export_sampled_points(self):
        # 导出所有"已选择点"菜单中的点的高维特征
        import pandas as pd
        page5 = self.controller.pages.get("page5")
        columns = [col for col in self.controller.data.columns if col not in ['MAE', 'MSE']]
        X = self.controller.data[columns].values
        hd_list = []
        n_raw = len(X)
        # 遍历所有已选择点
        for idx in self.selected_indices:
            if idx < n_raw:
                hd_list.append(X[idx])  # 原始点
            elif page5 is not None:
                hd_idx = idx - n_raw
                if 0 <= hd_idx < len(page5.new_points_hd):
                    hd_list.append(page5.new_points_hd[hd_idx])  # 补全点
        if not hd_list:
            tk.messagebox.showerror("Error", "没有可导出的已选择点")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件", "*.csv")])
        if file_path:
            df = pd.DataFrame(hd_list, columns=columns)
            df.to_csv(file_path, index=False)
            tk.messagebox.showinfo("导出成功", f"已选择点已导出到: {file_path}")
    def toggle_other_points(self):
        """切换其他点的显示状态"""
        self.hide_others.set(not self.hide_others.get())
        if self.hide_others.get():
            self.toggle_button.configure(text="Show Other Points")
        else:
            self.toggle_button.configure(text="Hide Other Points")
        self.update_plot()

class Page7(ttk.Frame):
    """
    功能来自 fig2jingxuan-fig2.py
    绘制性能曲线图
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.cases = []

        # --- UI ---
        # 控制面板
        control_frame = ttk.LabelFrame(self, text="Sampling Efficiency: Performance Curves")
        control_frame.pack(side="left", fill="y", padx=10, pady=10)

        # 列表框用于显示已添加的case
        self.case_list_frame = ttk.LabelFrame(control_frame, text="Cases")
        self.case_list_frame.pack(pady=5, padx=5, fill="both", expand=True)
        self.case_listbox = tk.Listbox(self.case_list_frame)
        self.case_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.case_list_frame, orient="vertical", command=self.case_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.case_listbox.config(yscrollcommand=scrollbar.set)

        # 按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5, padx=5, fill="x", side="bottom")
        ttk.Button(btn_frame, text="Add Case", command=self.add_case).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self.remove_case).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Run & Plot Curves", command=self.run_analysis).pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Export High‑Res Figures", command=self.export_plots).pack(fill="x", pady=2)

        # 绘图区 (可滚动)
        plot_container = ttk.Frame(self)
        plot_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(plot_container)
        scrollbar = ttk.Scrollbar(plot_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.figs = []

    def add_case(self):
        file_path = filedialog.askopenfilename(title="Select data file", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        label = tk.simpledialog.askstring("Input", "Enter a label for this case:", parent=self)
        if label:
            self.cases.append((file_path, label))
            self.case_listbox.insert(tk.END, f"{label}: {os.path.basename(file_path)}")

    def remove_case(self):
        selected_indices = self.case_listbox.curselection()
        if not selected_indices:
            return
        # 从后往前删，避免索引错乱
        for i in sorted(selected_indices, reverse=True):
            self.case_listbox.delete(i)
            del self.cases[i]

    def clear_all(self):
        self.cases.clear()
        self.case_listbox.delete(0, tk.END)
        for fig in self.figs:
            plt.close(fig)
        self.figs.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def run_analysis(self):
        if not self.cases:
            tk.messagebox.showwarning("Warning", "Please add at least one case.")
            return

        # 清理旧图
        for fig in self.figs:
            plt.close(fig)
        self.figs.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for csvfile, label in self.cases:
            try:
                fig = self.plot_single_case(csvfile, label)
                self.figs.append(fig)
                canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10, padx=10, fill="x", expand=True)
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to plot case '{label}':\n{e}")

    def plot_single_case(self, csvfile, label):
        # 使用 fig2jingxuan-fig2.py 的绘图风格和逻辑
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'legend.fontsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.edgecolor': '#222222',
            'axes.linewidth': 2.0,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'grid.color': '#e5e5e5',
            'grid.linewidth': 1.5,
            'legend.frameon': False,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.3,
            'font.family': 'Arial',
        })

        select_percents = [1, 5, 10, 15, 30, 40, 50, 60, 70, 80, 90, 100]
        
        df = pd.read_csv(csvfile)
        df_sel = df[df['percent'].isin(select_percents)].copy()
        mae = df_sel['mae'].values
        perc = df_sel['percent'].values
        time = df_sel['train_time'].values

        time_max, time_min = np.max(time), np.min(time)
        mae_max, mae_min = np.max(mae), np.min(mae)
        time_range, mae_range = time_max - time_min, mae_max - mae_min
        time_margin, mae_margin = time_range * 0.15, mae_range * 0.15

        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle(label + " — Sampling Efficiency", fontsize=20)
        ax2 = ax1.twinx()

        ax1.plot(perc, time, ':o', color='#1f77b4', lw=2.5, markersize=8, zorder=2, markeredgewidth=2, label="Train Time")
        ax2.plot(perc, mae, ':s', color='#d62728', lw=2.5, markersize=8, zorder=2, markeredgewidth=2, label="Test MAE (lower is better)")
        
        ax2.axvspan(0, 10, facecolor='#dddddd', alpha=0.3, zorder=0)

        ax1.set_xlabel('Sampling Percentage (%)', labelpad=10)
        ax1.set_ylabel('Train Time (s)', color='#1f77b4', labelpad=10)
        ax2.set_ylabel('Test MAE', color='#d62728', labelpad=10)
        
        ax1.tick_params(axis='y', colors='#1f77b4', pad=8, length=8, width=2)
        ax2.tick_params(axis='y', colors='#d62728', pad=8, length=8, width=2)
        ax1.tick_params(axis='x', pad=8, length=8, width=2)
        
        ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax1.set_xlim(-2, 102)

        ax1.set_ylim(time_min - time_margin, time_max * 1.5)
        ax2.set_ylim(mae_min - mae_margin, mae_max * 1.5)

        ax1.grid(True, which='major', axis='both', linestyle='--', linewidth=1.5, zorder=0)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
        fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make room for suptitle and legend
        return fig

    def export_plots(self):
        if not self.figs:
            tk.messagebox.showwarning("Warning", "No plots to export. Please run analysis first.")
            return
        
        output_dir = filedialog.askdirectory(title="Select directory to save plots")
        if not output_dir:
            return
            
        try:
            for i, fig in enumerate(self.figs):
                _, label = self.cases[i]
                safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '_')).rstrip()
                outname = os.path.join(output_dir, f"{safe_label}_highend.png")
                fig.savefig(outname, dpi=300, bbox_inches='tight')
            tk.messagebox.showinfo("Success", f"All plots exported to {output_dir}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to export plots:\n{e}")


class Page8(ttk.Frame):
    """
    功能来自 addpoint-fig3b.py
    交互式PCA可视化
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.df1, self.df3 = None, None
        self.feature_cols, self.is_new = None, None
        self.pca_result, self.explained_variance = None, None
        self.point_colors = None
        self.show_all_red = False
        self.selected_regions = []

        # --- UI ---
        main_layout = tk.Frame(self)
        main_layout.pack(fill="both", expand=True)

        # Top control panel
        top_panel = ttk.Frame(main_layout)
        top_panel.pack(side="top", fill="x", padx=5, pady=5)
        
        ttk.Button(top_panel, text="Load Original Data (1.csv)", command=self.load_df1).pack(side="left", padx=5)
        ttk.Button(top_panel, text="Load Supplementary Data (3.csv)", command=self.load_df3).pack(side="left", padx=5)
        ttk.Button(top_panel, text="Run PCA Analysis", command=self.run_pca).pack(side="left", padx=5)
        # 将 Export Figure 移到顶部并列显示
        self.export_btn = ttk.Button(top_panel, text="Export Figure", command=self.export_figure, state="disabled")
        self.export_btn.pack(side="left", padx=5)
        
        self.info_label = ttk.Label(top_panel, text="Please load data and run analysis.")
        self.info_label.pack(side="left", padx=10)

        # 删除底部按钮面板（精简页面）

        # Plot area
        self.plot_frame = ttk.Frame(main_layout)
        self.plot_frame.pack(side="top", fill="both", expand=True)
        self.fig = plt.Figure(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_df1(self):
        path = filedialog.askopenfilename(title="Load Original Data", filetypes=[("CSV", "*.csv")])
        if path:
            self.df1 = pd.read_csv(path)
            tk.messagebox.showinfo("Success", f"Loaded {os.path.basename(path)} as original data.")

    def load_df3(self):
        path = filedialog.askopenfilename(title="Load Supplementary Data", filetypes=[("CSV", "*.csv")])
        if path:
            self.df3 = pd.read_csv(path)
            tk.messagebox.showinfo("Success", f"Loaded {os.path.basename(path)} as supplementary data.")

    def run_pca(self):
        if self.df1 is None or self.df3 is None:
            tk.messagebox.showerror("Error", "Please load both data files first.")
            return
        try:
            # 仅使用两个文件的公共特征列
            self.feature_cols = [c for c in self.df1.columns.tolist() if c in self.df3.columns.tolist()]
            if not self.feature_cols:
                raise ValueError("No common feature columns between two files.")

            # 合并后统一标准化与PCA，随后分割回两组
            combined = pd.concat([self.df1[self.feature_cols], self.df3[self.feature_cols]], ignore_index=True)
            scaler = StandardScaler()
            combined_scaled = scaler.fit_transform(combined)
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined_scaled)
            self.explained_variance = pca.explained_variance_ratio_

            n1 = len(self.df1)
            self.pca_result1 = combined_pca[:n1]
            self.pca_result3 = combined_pca[n1:]

            # 为了兼容现有绘制函数，仍设置总结果和标记
            self.pca_result = combined_pca
            self.is_new = np.array([False]*n1 + [True]*len(self.df3))

            self.info_label.config(text=f"Explained variance: PC1 {self.explained_variance[0]:.2%}, PC2 {self.explained_variance[1]:.2%}. Original: {n1}, Supplementary: {len(self.df3)}")

            # 只开启导出图按钮
            if hasattr(self, 'export_btn'):
                self.export_btn.config(state="normal")

            self.plot_all()
        except Exception as e:
            tk.messagebox.showerror("PCA Error", f"Failed to perform PCA: {e}")

    def plot_all(self):
        self.fig.clf()
        self.ax = self.fig.add_axes([0.12, 0.12, 0.64, 0.75])

        # 直接显示原始点与补充点两类（先画红色补充点，再画蓝色原始点，避免被覆盖）
        self.ax.scatter(self.pca_result3[:, 0], self.pca_result3[:, 1], c='#FF3333', label='Supplementary Points',
                        alpha=0.38, s=34, marker='^', edgecolors='none', zorder=2)
        self.ax.scatter(self.pca_result1[:, 0], self.pca_result1[:, 1], c='#66B2FF', label='Original Dataset',
                        alpha=0.85, s=38, marker='o', edgecolors='k', linewidths=0.4, zorder=3)

        self.ax.set_title('PCA of Datasets', fontsize=16, pad=20)
        self.ax.set_xlabel('Dimension 1', fontsize=12)
        self.ax.set_ylabel('Dimension 2', fontsize=12)
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)

        self.canvas.draw()

    def activate_selector(self):
        self.selector = RectangleSelector(self.ax, self.onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        if hasattr(self, 'focus_btn'):
            self.focus_btn.config(state="disabled")

    def onselect(self, eclick, erelease):
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        self.selected_regions.append((x_min, x_max, y_min, y_max))
        self.plot_all()
        self.selector.set_active(False)
        if hasattr(self, 'focus_btn'):
            self.focus_btn.config(state="normal")
    
    def clear_selections(self):
        self.selected_regions.clear()
        self.plot_all()

    def toggle_show_all(self):
        self.show_all_red = not self.show_all_red
        if self.show_all_red:
            if hasattr(self, 'show_all_btn'):
                self.show_all_btn.config(text='Show Selection Only')
            if self.point_colors is None:
                self.point_colors = self.is_new.copy()
        else:
            if hasattr(self, 'show_all_btn'):
                self.show_all_btn.config(text='Show All New Points')
        self.plot_all()

    def export_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            self.fig.savefig(path, dpi=300)
            tk.messagebox.showinfo("Success", f"Figure saved to {path}")

    def export_updated_file1(self):
        if self.point_colors is None:
            tk.messagebox.showerror("Error", "Point colors not initialized. Please switch to 'Show All New Points' mode first.")
            return
            
        blue_points_mask = ~self.point_colors
        updated_df = self.df3[blue_points_mask]
        
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            updated_df.to_csv(path, index=False)
            tk.messagebox.showinfo("Success", f"Updated data saved to {path}")

class Page9(ttk.Frame):
    """
    功能来自 addplot-fig3.py
    增量学习分析
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Data
        self.df_hole, self.df_test, self.df_full = None, None, None

        # --- UI ---
        control_frame = ttk.LabelFrame(self, text="Exploration Impact: Incremental Learning")
        control_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load Original Training Data", command=self.load_hole).pack(fill="x", pady=5)
        self.hole_label = ttk.Label(control_frame, text="Not loaded", foreground="red")
        self.hole_label.pack(fill="x", padx=5)

        ttk.Button(control_frame, text="Load Test Data", command=self.load_test).pack(fill="x", pady=5)
        self.test_label = ttk.Label(control_frame, text="Not loaded", foreground="red")
        self.test_label.pack(fill="x", padx=5)
        
        ttk.Button(control_frame, text="Load Augmented Training Data", command=self.load_full).pack(fill="x", pady=5)
        self.full_label = ttk.Label(control_frame, text="Not loaded", foreground="red")
        self.full_label.pack(fill="x", padx=5)

        ttk.Button(control_frame, text="Run Exploration Impact Analysis", command=self.run_analysis).pack(fill="x", pady=20)
        
        # Plot area
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side="right", fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_hole(self):
        path = filedialog.askopenfilename(title="Load Hole Data", filetypes=[("CSV", "*.csv")])
        if path:
            self.df_hole = pd.read_csv(path)
            self.hole_label.config(text=os.path.basename(path), foreground="green")

    def load_test(self):
        path = filedialog.askopenfilename(title="Load Test Data", filetypes=[("CSV", "*.csv")])
        if path:
            self.df_test = pd.read_csv(path)
            self.test_label.config(text=os.path.basename(path), foreground="green")

    def load_full(self):
        path = filedialog.askopenfilename(title="Load Full Data", filetypes=[("CSV", "*.csv")])
        if path:
            self.df_full = pd.read_csv(path)
            self.full_label.config(text=os.path.basename(path), foreground="green")

    def run_analysis(self):
        if self.df_hole is None or self.df_test is None or self.df_full is None:
            tk.messagebox.showerror("Error", "Please load all three data files.")
            return
        if XGBRegressor is None:
            tk.messagebox.showerror("Error", "xgboost library is not installed. Please install it (`pip install xgboost`)")
            return
            
        progress_window = tk.Toplevel(self)
        progress_window.title("Analysis in Progress")
        label = ttk.Label(progress_window, text="Analysis started, this may take a while...")
        label.pack(padx=20, pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', length=300)
        progress_bar.pack(padx=20, pady=10)
        progress_window.grab_set()

        thread = threading.Thread(target=self._analysis_thread, args=(progress_window, progress_bar, label))
        thread.start()

    def _analysis_thread(self, progress_window, progress_bar, label):
        try:
            df_full, df_hole, df_test = self.df_full.copy(), self.df_hole.copy(), self.df_test.copy()
            
            feature_cols = list(df_full.columns[:-1])
            target_col = df_full.columns[-1]

            combined = pd.concat([df_full[feature_cols], df_hole[feature_cols]])
            is_duplicated = combined.duplicated(keep=False)
            unique_to_full_mask = ~is_duplicated.iloc[:len(df_full)]
            wait_add_df = df_full[unique_to_full_mask].reset_index(drop=True)
            
            mae_list = []
            n_points = []
            
            X_test = np.asarray(df_test[feature_cols].values, dtype=np.float32)
            y_test = np.asarray(df_test[target_col].values, dtype=np.float32)
            
            def fillna(X, ref):
                X_filled = np.nan_to_num(X, nan=np.nanmean(ref, axis=0))
                return X_filled

            model = XGBRegressor(max_depth=5, n_estimators=200, n_jobs=-1, eval_metric="mae", tree_method='hist')

            total_to_add = len(wait_add_df)
            batch_num = 20
            all_dfs_to_process = [df_hole] + [pd.concat([df_hole, wait_add_df.iloc[:end]], ignore_index=True) for end in np.linspace(0, total_to_add, batch_num, dtype=int)[1:]]
            all_dfs_to_process.append(pd.concat([df_hole, wait_add_df], ignore_index=True))
            
            total_steps = len(all_dfs_to_process)
            for i, df_new_train in enumerate(all_dfs_to_process):
                self.after_idle(label.config, {'text': f"Processing step {i+1}/{total_steps}"})
                self.after_idle(progress_bar.config, {'value': (i+1)/total_steps * 100})

                X_train = np.asarray(df_new_train[feature_cols].values, dtype=np.float32)
                y_train = np.asarray(df_new_train[target_col].values, dtype=np.float32)
                X_train_filled = fillna(X_train, X_train)
                X_test_filled = fillna(X_test, np.vstack([X_train, X_test]))
                
                model.fit(X_train_filled, y_train)
                y_pred = model.predict(X_test_filled)
                abs_err = np.abs(y_pred - y_test)
                
                mae_list.append(abs_err)
                n_points.append(len(df_new_train) - len(df_hole))
            
            self.after_idle(self.plot_results, mae_list, n_points, total_to_add)
        except Exception as e:
            self.after_idle(tk.messagebox.showerror, "Error", f"Analysis failed: {e}")
        finally:
            self.after_idle(progress_window.destroy)

    def plot_results(self, mae_list, n_points, total_added):
        plt.rcParams.update({
            'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
            'legend.fontsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10
        })
        self.ax.clear()
        
        percent_list = [n / total_added * 100 if total_added > 0 else 0 for n in n_points]
        means = [np.mean(mae) for mae in mae_list]

        self.ax.boxplot(mae_list, positions=percent_list, widths=3, showfliers=False)
        self.ax.set_xlabel('Percentage of Added Points (%)')
        self.ax.set_ylabel('Test Set Absolute Error Distribution (MAE)')
        
        ax2 = self.ax.twinx()
        ax2.plot(percent_list, means, '-o', color='#d62728', label='Mean MAE')
        ax2.set_ylabel('Mean MAE')

        self.ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        
        self.ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        self.fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        self.ax.set_title('Test Set Error vs. Added Points')
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop() 
