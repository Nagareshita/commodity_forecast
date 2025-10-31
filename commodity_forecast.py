"""
コモディティ価格予測ツール（修正版）
Commodity Price Forecasting Tool

CSV形式のデータから将来価格を予測
"""

import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QProgressBar,
                               QSpinBox, QGroupBox, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（フォールバック付き）
import matplotlib.font_manager as fm

# 利用可能なフォントを確認
available_fonts = [f.name for f in fm.fontManager.ttflist]
japanese_fonts = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'Hiragino Sans', 
                  'Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']

# 利用可能な日本語フォントを探す
font_to_use = 'DejaVu Sans'  # デフォルト
for font in japanese_fonts:
    if font in available_fonts:
        font_to_use = font
        break

plt.rcParams['font.family'] = font_to_use
plt.rcParams['font.sans-serif'] = [font_to_use]
plt.rcParams['axes.unicode_minus'] = False

# 学術的な白背景スタイル
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'


class CommodityForecaster:
    """コモディティ価格予測クラス（修正版）"""
    
    def __init__(self, data):
        self.data = data
        self.prices = data['Price'].values
        self.dates = data['Date']
        
        # データ頻度の自動検出
        time_diff = (self.dates.iloc[1] - self.dates.iloc[0]).days
        if time_diff >= 20:  # 月次データ
            self.periods_per_year = 12
            self.freq_name = "Monthly"
        elif time_diff >= 5:  # 週次データ
            self.periods_per_year = 52
            self.freq_name = "Weekly"
        else:  # 日次データ
            self.periods_per_year = 252
            self.freq_name = "Daily"
        
        # 統計量の計算（修正版）
        returns = np.diff(np.log(self.prices))
        # 期間リターンの平均・標準偏差
        self.period_mu = np.mean(returns)
        self.period_sigma = np.std(returns)
        
        # 年率換算
        self.annual_mu = self.period_mu * self.periods_per_year
        self.annual_sigma = self.period_sigma * np.sqrt(self.periods_per_year)
        
        print(f"データ頻度: {self.freq_name} ({self.periods_per_year} periods/year)")
        print(f"統計量:")
        print(f"  期間リターン平均: {self.period_mu:.6f}")
        print(f"  期間ボラティリティ: {self.period_sigma:.6f}")
        print(f"  年率リターン: {self.annual_mu:.4f} ({self.annual_mu*100:.2f}%)")
        print(f"  年率ボラティリティ: {self.annual_sigma:.4f} ({self.annual_sigma*100:.2f}%)")
    
    def forecast(self, years=30, n_simulations=1000):
        """
        Combined Model による予測（修正版）
        """
        print(f"\n予測実行中 (Combined Model)...")
        
        # ARIMA予測（トレンド成分のみ使用）
        print("  ARIMA計算中...")
        try:
            model = ARIMA(self.prices, order=(1, 1, 1))  # よりシンプルなパラメータ
            fitted_model = model.fit()
            
            # 短期予測のみ（1年分）
            n_periods_data = years * self.periods_per_year
            arima_steps = min(self.periods_per_year, n_periods_data)
            arima_forecast = fitted_model.forecast(steps=arima_steps)
            
            if isinstance(arima_forecast, np.ndarray):
                arima_values = arima_forecast
            else:
                arima_values = arima_forecast.values
            
            # ARIMAのトレンド率を計算
            if len(arima_values) > 0:
                arima_trend_rate = (arima_values[-1] / arima_values[0]) ** (1/len(arima_values))
                arima_annual_rate = (arima_trend_rate ** self.periods_per_year) - 1
                print(f"  ARIMA年率トレンド: {arima_annual_rate:.4f} ({arima_annual_rate*100:.2f}%)")
            else:
                arima_annual_rate = 0
        except:
            print("  ARIMA計算をスキップ（データ不足）")
            arima_annual_rate = 0
        
        # モンテカルロシミュレーション
        print(f"  モンテカルロシミュレーション中 ({n_simulations}回)...")
        last_price = self.prices[-1]
        n_periods = years * self.periods_per_year
        dt = 1/self.periods_per_year
        
        # Combined Modelのパラメータ設定
        # ARIMAトレンドとヒストリカルトレンドをブレンド
        combined_mu = 0.7 * self.annual_mu + 0.3 * arima_annual_rate
        combined_sigma = self.annual_sigma
        
        print(f"  予測パラメータ:")
        print(f"    年率ドリフト: {combined_mu:.4f} ({combined_mu*100:.2f}%)")
        print(f"    年率ボラティリティ: {combined_sigma:.4f} ({combined_sigma*100:.2f}%)")
        
        simulations = np.zeros((n_simulations, n_periods))
        
        for i in range(n_simulations):
            prices = np.zeros(n_periods)
            prices[0] = last_price
            
            # Geometric Brownian Motionの実装（修正版）
            for t in range(1, n_periods):
                shock = np.random.normal(0, 1)
                
                # 修正版: 正しいGBM公式
                drift = (combined_mu - 0.5 * combined_sigma**2) * dt
                diffusion = combined_sigma * np.sqrt(dt) * shock
                
                prices[t] = prices[t-1] * np.exp(drift + diffusion)
            
            simulations[i] = prices
        
        # 表示用の間引き（月次相当に）
        if self.periods_per_year == 12:
            # 既に月次なので間引き不要
            display_indices = np.arange(len(simulations[0]))
        else:
            # 月次相当に間引き
            step = max(1, self.periods_per_year // 12)
            display_indices = np.arange(0, n_periods, step)
        
        display_simulations = simulations[:, display_indices]
        
        # 統計量の計算
        mean_forecast = np.mean(display_simulations, axis=0)
        lower_90 = np.percentile(display_simulations, 5, axis=0)
        upper_90 = np.percentile(display_simulations, 95, axis=0)
        
        # 日付生成
        last_date = self.dates.iloc[-1]
        if self.periods_per_year == 12:
            freq = 'ME'
        elif self.periods_per_year == 52:
            freq = '7D'
        else:
            freq = '21D'
        
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=int(365/self.periods_per_year)),
            periods=len(display_indices),
            freq=freq
        )
        
        print("  予測完了！")
        print(f"\n予測結果:")
        print(f"  開始価格: {last_price:.2f} JPY/kg")
        print(f"  {years}年後予測: {mean_forecast[-1]:.2f} JPY/kg")
        print(f"  倍率: {mean_forecast[-1]/last_price:.2f}x")
        print(f"  年平均成長率: {((mean_forecast[-1]/last_price)**(1/years) - 1)*100:.2f}%")
        print()
        
        return {
            'dates': forecast_dates,
            'mean': mean_forecast,
            'lower_90': lower_90,
            'upper_90': upper_90
        }


class ForecastThread(QThread):
    """予測計算用スレッド"""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, data, forecast_years):
        super().__init__()
        self.data = data
        self.forecast_years = forecast_years
        
    def run(self):
        try:
            self.progress.emit(20)
            forecaster = CommodityForecaster(self.data)
            self.progress.emit(40)
            result = forecaster.forecast(years=self.forecast_years)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class MplCanvas(FigureCanvas):
    """Matplotlibキャンバス"""
    def __init__(self, parent=None, width=12, height=7, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
        self.axes = fig.add_subplot(111, facecolor='white')
        super().__init__(fig)


class CommodityForecastApp(QMainWindow):
    """メインアプリケーション"""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.forecast_result = None
        self.commodity_name = ""
        self.init_ui()
        
    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle("Commodity Price Forecasting Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # スタイルシート（白背景）
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: white;
                color: black;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #2c5aa0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3a6fb8;
            }
            QPushButton:pressed {
                background-color: #1e4278;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                color: black;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #f5f5f5;
                color: black;
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #2c5aa0;
                border-radius: 5px;
            }
            QGroupBox {
                border: 2px solid #cccccc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #2c5aa0;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # タイトル
        title_label = QLabel("Commodity Price Forecasting Tool")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c5aa0; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # コントロールパネル
        control_group = QGroupBox("Settings")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)
        
        # 予測期間
        forecast_label = QLabel("Forecast Period:")
        self.forecast_years_spin = QSpinBox()
        self.forecast_years_spin.setRange(5, 50)
        self.forecast_years_spin.setValue(30)
        self.forecast_years_spin.setSuffix(" years")
        
        control_layout.addWidget(forecast_label)
        control_layout.addWidget(self.forecast_years_spin)
        control_layout.addStretch()
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # ボタンパネル
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.template_btn = QPushButton("Export CSV Template")
        self.template_btn.clicked.connect(self.export_template)
        
        self.load_btn = QPushButton("Load CSV File")
        self.load_btn.clicked.connect(self.load_csv)
        
        self.forecast_btn = QPushButton("Run Forecast")
        self.forecast_btn.clicked.connect(self.run_forecast)
        self.forecast_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        
        button_layout.addWidget(self.template_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.forecast_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # ステータス
        self.status_label = QLabel("Please load a CSV file")
        self.status_label.setStyleSheet("color: #2c5aa0; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        # グラフ
        self.canvas = MplCanvas(self, width=12, height=7, dpi=100)
        main_layout.addWidget(self.canvas)

        # インタラクション用の状態
        self.hover_hline = None
        self.hover_vline = None
        self.hover_text = None
        self.permanent_points = []  # [(point_artist, text_artist), ...]

        # マウスイベント接続
        self.cid_move = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        self.show_welcome()

    def _init_interaction_artists(self):
        """クロスヘアとホバーテキストを初期化（axesをクリア後に再生成）"""
        ax = self.canvas.axes
        # 既存があれば無視（axes.clear()後は消えているので参照は無効）
        self.hover_hline = ax.axhline(y=ax.get_ylim()[0], color='#888888', lw=0.8, ls='--', alpha=0.6)
        self.hover_vline = ax.axvline(x=ax.get_xlim()[0], color='#888888', lw=0.8, ls='--', alpha=0.6)
        self.hover_hline.set_visible(False)
        self.hover_vline.set_visible(False)
        # カーソル付近に値を表示
        self.hover_text = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", alpha=0.9),
            fontsize=9,
        )
        self.hover_text.set_visible(False)
        
    def show_welcome(self):
        """初期画面"""
        self.canvas.axes.clear()
        self.canvas.axes.text(
            0.5, 0.5, 
            'Commodity Price Forecasting Tool\n\nPlease load a CSV file',
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.canvas.axes.transAxes,
            fontsize=16,
            color='#2c5aa0',
            fontweight='bold'
        )
        self.canvas.axes.set_xticks([])
        self.canvas.axes.set_yticks([])
        self.canvas.draw()
        # 画面初期化時はインタラクション用アーティストを作らない
    
    def export_template(self):
        """CSVひな型を出力"""
        try:
            # サンプルデータ生成
            start_date = datetime(2000, 1, 1)
            dates = pd.date_range(start=start_date, periods=100, freq='ME')
            
            # サンプル価格（JPY/kg）- より現実的な変動
            base_price = 1000
            trend = np.linspace(0, 200, 100)  # 緩やかな上昇トレンド
            cycle = 100 * np.sin(2 * np.pi * np.arange(100) / 24)  # 周期変動
            noise = np.random.randn(100) * 30  # ノイズ
            prices = base_price + trend + cycle + noise
            prices = np.maximum(prices, 500)  # 最小値
            
            df = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Price': prices.round(2)
            })
            
            output_path = '/mnt/user-data/outputs/commodity_template.csv'
            df.to_csv(output_path, index=False)
            
            self.status_label.setText(f"Template exported: commodity_template.csv")
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"CSV template has been exported.\n\n"
                f"File: commodity_template.csv\n\n"
                f"Format:\n"
                f"Date,Price\n"
                f"2000-01-01,1000.00\n"
                f"2000-02-01,1050.50\n"
                f"...\n\n"
                f"Note: Price should be in JPY/kg"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Template export error:\n{str(e)}")
    
    def load_csv(self):
        """CSVファイル読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # CSV読み込み
            df = pd.read_csv(file_path)
            
            # カラム確認
            if 'Date' not in df.columns or 'Price' not in df.columns:
                raise ValueError("CSV must contain 'Date' and 'Price' columns")
            
            # 日付変換
            df['Date'] = pd.to_datetime(df['Date'])
            
            # ソート
            df = df.sort_values('Date').reset_index(drop=True)
            
            # データ保存
            self.data = df
            self.commodity_name = os.path.basename(file_path).replace('.csv', '')
            
            # 統計表示
            self.status_label.setText(
                f"Loaded: {len(df)} data points "
                f"({df['Date'].min().date()} - {df['Date'].max().date()})"
            )
            
            self.forecast_btn.setEnabled(True)
            
            # グラフ表示
            self.plot_historical()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"CSV loading error:\n{str(e)}")
    
    def plot_historical(self):
        """過去データ表示"""
        self.canvas.axes.clear()
        
        self.canvas.axes.plot(
            self.data['Date'], 
            self.data['Price'], 
            color='#2c5aa0', 
            linewidth=2,
            label='Historical Data'
        )
        
        self.canvas.axes.set_xlabel('Year', fontsize=12, fontweight='bold')
        self.canvas.axes.set_ylabel('Price (JPY/kg)', fontsize=12, fontweight='bold')
        self.canvas.axes.set_title(
            f'{self.commodity_name} - Price History',
            fontsize=14,
            fontweight='bold',
            color='#2c5aa0',
            pad=15
        )
        self.canvas.axes.legend(loc='best', frameon=True, shadow=True)
        self.canvas.axes.grid(True, alpha=0.5, linestyle='--')
        
        self.canvas.figure.tight_layout()
        # クリックで追加した点は新しい描画でリセット
        self.permanent_points = []
        # ホバー用アーティストを再作成
        self._init_interaction_artists()
        self.canvas.draw()
    
    def run_forecast(self):
        """予測実行"""
        if self.data is None:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Forecasting...")
        self.forecast_btn.setEnabled(False)
        
        years = self.forecast_years_spin.value()
        
        self.forecast_thread = ForecastThread(self.data, years)
        self.forecast_thread.progress.connect(self.progress_bar.setValue)
        self.forecast_thread.finished.connect(self.on_forecast_finished)
        self.forecast_thread.error.connect(self.on_error)
        self.forecast_thread.start()
    
    def on_forecast_finished(self, result):
        """予測完了"""
        self.forecast_result = result
        self.progress_bar.setVisible(False)
        self.forecast_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.status_label.setText("Forecast completed")
        
        self.plot_forecast()
    
    def plot_forecast(self):
        """予測結果表示"""
        self.canvas.axes.clear()
        
        # 過去データ
        self.canvas.axes.plot(
            self.data['Date'],
            self.data['Price'],
            color='#2c5aa0',
            linewidth=2.5,
            label='Historical Data',
            zorder=3
        )
        
        # 予測
        forecast_dates = self.forecast_result['dates']
        forecast_mean = self.forecast_result['mean']
        lower = self.forecast_result['lower_90']
        upper = self.forecast_result['upper_90']
        
        self.canvas.axes.plot(
            forecast_dates,
            forecast_mean,
            color='#d62728',
            linewidth=2.5,
            label='Forecast Mean',
            linestyle='--',
            zorder=2
        )
        
        self.canvas.axes.fill_between(
            forecast_dates,
            lower,
            upper,
            color='#d62728',
            alpha=0.2,
            label='90% Confidence Interval',
            zorder=1
        )
        
        # 接続線
        self.canvas.axes.plot(
            [self.data['Date'].iloc[-1], forecast_dates[0]],
            [self.data['Price'].iloc[-1], forecast_mean[0]],
            color='gray',
            linestyle=':',
            linewidth=1.5,
            alpha=0.6,
            zorder=1
        )
        
        self.canvas.axes.set_xlabel('Year', fontsize=12, fontweight='bold')
        self.canvas.axes.set_ylabel('Price (JPY/kg)', fontsize=12, fontweight='bold')
        self.canvas.axes.set_title(
            f'{self.commodity_name} - {self.forecast_years_spin.value()}-Year Forecast (Combined Model)',
            fontsize=14,
            fontweight='bold',
            color='#2c5aa0',
            pad=15
        )
        self.canvas.axes.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        self.canvas.axes.grid(True, alpha=0.5, linestyle='--')
        
        self.canvas.figure.tight_layout()
        # クリックで追加した点は新しい描画でリセット
        self.permanent_points = []
        # ホバー用アーティストを再作成
        self._init_interaction_artists()
        self.canvas.draw()
    
    def export_results(self):
        """結果をCSV出力"""
        if self.forecast_result is None:
            return
        
        try:
            # 予測データ
            forecast_df = pd.DataFrame({
                'Date': self.forecast_result['dates'].strftime('%Y-%m-%d'),
                'Forecast_Mean': self.forecast_result['mean'].round(2),
                'Lower_90': self.forecast_result['lower_90'].round(2),
                'Upper_90': self.forecast_result['upper_90'].round(2)
            })
            
            # 過去データ
            hist_df = self.data.copy()
            hist_df['Date'] = hist_df['Date'].dt.strftime('%Y-%m-%d')
            hist_df['Price'] = hist_df['Price'].round(2)
            
            # 保存
            base_name = self.commodity_name
            hist_path = f'/mnt/user-data/outputs/{base_name}_historical.csv'
            forecast_path = f'/mnt/user-data/outputs/{base_name}_forecast.csv'
            
            hist_df.to_csv(hist_path, index=False)
            forecast_df.to_csv(forecast_path, index=False)
            
            self.status_label.setText("Results exported")
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Files saved:\n\n"
                f"- {base_name}_historical.csv (Historical Data)\n"
                f"- {base_name}_forecast.csv (Forecast Data)"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export error:\n{str(e)}")
    
    def on_error(self, error_msg):
        """エラー処理"""
        self.progress_bar.setVisible(False)
        self.forecast_btn.setEnabled(True)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", error_msg)

    # ===== インタラクション（ホバー表示とクリック固定）=====
    def on_mouse_move(self, event):
        """マウス移動で軸値を表示（クロスヘア + ツールチップ）"""
        if event.inaxes != self.canvas.axes or event.xdata is None or event.ydata is None:
            # 領域外なら非表示
            if self.hover_hline is not None:
                self.hover_hline.set_visible(False)
            if self.hover_vline is not None:
                self.hover_vline.set_visible(False)
            if self.hover_text is not None:
                self.hover_text.set_visible(False)
            self.canvas.draw_idle()
            return

        ax = self.canvas.axes
        x = event.xdata
        y = event.ydata

        # X（日付）フォーマット
        try:
            dt = mdates.num2date(x)
            x_str = dt.strftime('%Y-%m-%d')
        except Exception:
            x_str = f"{x:.3f}"
        y_str = f"{y:,.2f}"

        # クロスヘア更新
        if self.hover_hline is None or self.hover_vline is None or self.hover_text is None:
            self._init_interaction_artists()
        self.hover_hline.set_ydata([y])
        self.hover_vline.set_xdata([x])
        self.hover_hline.set_visible(True)
        self.hover_vline.set_visible(True)

        # テキスト更新（カーソルの少し右上）
        self.hover_text.xy = (x, y)
        self.hover_text.set_text(f"X: {x_str}\nY: {y_str}")
        self.hover_text.set_visible(True)

        self.canvas.draw_idle()

    def on_mouse_click(self, event):
        """クリックでその点を固定表示（マーカー + ラベル）"""
        if event.button != 1:  # 左クリックのみ
            return
        if event.inaxes != self.canvas.axes or event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata

        # 表示テキスト
        try:
            dt = mdates.num2date(x)
            x_str = dt.strftime('%Y-%m-%d')
        except Exception:
            x_str = f"{x:.3f}"
        y_str = f"{y:,.2f}"

        # マーカーと注釈を追加
        pt = self.canvas.axes.plot(x, y, marker='o', color='#ff7f0e', markersize=6, zorder=6)[0]
        txt = self.canvas.axes.annotate(
            f"{x_str}\n{y_str}",
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ff7f0e", alpha=0.95),
            fontsize=9,
        )
        self.permanent_points.append((pt, txt))
        self.canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    window = CommodityForecastApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
