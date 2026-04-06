import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class DistortionVisualizer:
    def __init__(self, yaml_path, resolution=(1920, 1080)):
        self.width, self.height = resolution
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }
        self.read_yaml_file(yaml_path)
        
    def read_yaml_file(self, yaml_file):
        """
        Read YAML file to extract cameras parameters
        """
        # Load the YAML file
        with open(yaml_file) as file:  # Replace with your file path
            params = yaml.safe_load(file)

            # Parse the matrices
        self.camera_params['left']['kk'] = np.array(params['camera_matrix_left'], dtype=np.float64)
        self.camera_params['left']['kc'] = np.array(params['dist_coeffs_left'], dtype=np.float64)
        self.camera_params['left']['r'] = np.array(params['rot_matrix_left'], dtype=np.float64)
        self.camera_params['left']['t'] = np.array(params['t_left'], dtype=np.float64)

        self.camera_params['right']['kk'] = np.array(params['camera_matrix_right'], dtype=np.float64)
        self.camera_params['right']['kc'] = np.array(params['dist_coeffs_right'], dtype=np.float64)
        self.camera_params['right']['r'] = np.array(params['rot_matrix_right'], dtype=np.float64)
        self.camera_params['right']['t'] = np.array(params['t_right'], dtype=np.float64)

        self.camera_params['stereo']['R'] = np.array(params['R'], dtype=np.float64)
        self.camera_params['stereo']['T'] = np.array(params['T'], dtype=np.float64)

    def generate_grid_points(self, step=50):
        """Creates a grid of points across the image sensor."""
        x = np.arange(0, self.width, step)
        y = np.arange(0, self.height, step)
        return np.meshgrid(x, y)

    def compute_distortion_field(self, camera='left', mode='all'):
        """
        Calculates the displacement of points.
        Modes: 'radial', 'tangential', 'all'
        """
        u, v = self.generate_grid_points()
        points = np.stack([u.flatten(), v.flatten()], axis=1).astype(np.float32)
        
        # k1, k2, p1, p2, k3...
        d = self.camera_params[camera]['kc'].flatten()
        
        if mode == 'radial':
            # Zero out p1, p2 (indices 2 and 3)
            d_mod = np.array([d[0], d[1], 0, 0, d[4] if len(d)>4 else 0])
        elif mode == 'tangential':
            # Zero out k1, k2, k3
            d_mod = np.array([0, 0, d[2], d[3], 0])
        else:
            d_mod = d

        # Use undistortPoints to see where the distorted points would 'move' to
        # or projectPoints to manually apply the forward distortion model.
        # Here we manually apply the model for better control of the plot.
        fx, fy = self.camera_params[camera]['kk'][0, 0], self.camera_params[camera]['kk'][1, 1]
        cx, cy = self.camera_params[camera]['kk'][0, 2], self.camera_params[camera]['kk'][1, 2]

        # Normalized coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        r2 = x**2 + y**2
        
        # Radial component
        k1, k2, p1, p2, k3 = d_mod[:5]
        radial = (1 + k1*r2 + k2*r2**2 + k3*r2**3)
        
        # Tangential component
        dx_tan = 2*p1*x*y + p2*(r2 + 2*x**2)
        dy_tan = p1*(r2 + 2*y**2) + 2*p2*x*y
        
        u_dist = fx * (x * radial + dx_tan) + cx
        v_dist = fy * (y * radial + dy_tan) + cy
        
        return u, v, u_dist, v_dist

    def plot_distortion(self, mode='all', save_path=None):
        """
        Plots left and right camera distortion side-by-side with color-coded magnitude.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        sides = ['left', 'right']
        
        # Guardar o objeto quiver para criar a colorbar depois
        last_q = None

        for i, side in enumerate(sides):
            u, v, u_dist, v_dist = self.compute_distortion_field(camera=side, mode=mode)
            
            # Calcular a magnitude do erro (em pixels)
            du = u_dist - u
            dv = v_dist - v
            magnitudes = np.sqrt(du**2 + dv**2)
            
            ax = axes[i]
            
            # Quiver com escala de cor (cmap)
            # pivot='mid' ajuda a centralizar a origem do vetor no ponto do grid
            q = ax.quiver(u, v, du, dv, magnitudes,
                          cmap='viridis', # 'inferno', 'magma' ou 'plasma' também são ótimos
                          scale_units='xy', 
                          scale=0.1, 
                          alpha=0.9)
            
            ax.set_title(f"{side.upper()} Camera - {mode.capitalize()} Distortion", fontsize=15, pad=15)
            ax.invert_yaxis()
            ax.set_xlabel("Pixels (X)")
            if i == 0: ax.set_ylabel("Pixels (Y)")
            ax.grid(True, linestyle=':', alpha=0.3)
            last_q = q

        # --- Adicionar Colorbar ---
        # Criamos uma barra lateral para indicar o erro em pixels
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5]) # [left, bottom, width, height]
        cbar = fig.colorbar(last_q, cax=cbar_ax)
        cbar.set_label('Distortion Magnitude (Pixels)', rotation=270, labelpad=15)

        # --- Stereo Metadata ---
        T = self.camera_params['stereo']['T']
        baseline = np.linalg.norm(T)
        
        # Rotação Simplificada (Euler)
        R = self.camera_params['stereo']['R']
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        y_angle = np.degrees(np.arctan2(-R[2,0], sy))

        stats_text = (
            f"Stereo Baseline: {baseline:.4f}\n"
            f"Relative Y-Rotation (Yaw): {y_angle:.3f}°"
        )
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.6'))

        # Ajuste fino para não sobrepor a colorbar
        plt.subplots_adjust(left=0.05, right=0.9, bottom=0.15, top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot salvo com escala de cor em: {save_path}")
            
        plt.show()

    def compute_field(self, camera='left', mode='radial', step=40):
        """ Gera a malha e calcula o deslocamento para o componente específico """
        x = np.arange(0, self.width + step, step)
        y = np.arange(0, self.height + step, step)
        u, v = np.meshgrid(x, y)
        
        p = self.camera_params[camera]
        fx, fy = p['kk'][0, 0], p['kk'][1, 1]
        cx, cy = p['kk'][0, 2], p['kk'][1, 2]
        d = p['kc'].flatten()

        # Coordenadas normalizadas
        xn = (u - cx) / fx
        yn = (v - cy) / fy
        r2 = xn**2 + yn**2
        r4 = r2**2
        r6 = r2**3
        
        # k1, k2, p1, p2, k3
        k1, k2, p1, p2, k3 = d[0], d[1], d[2], d[3], d[4] if len(d)>4 else 0

        if mode == 'radial':
            # Apenas componente radial
            radial_scale = (1 + k1*r2 + k2*r4 + k3*r6)
            u_dist = fx * (xn * radial_scale) + cx
            v_dist = fy * (yn * radial_scale) + cy
        else:
            # Apenas componente tangencial
            dx_tan = 2*p1*xn*yn + p2*(r2 + 2*xn**2)
            dy_tan = p1*(r2 + 2*yn**2) + 2*p2*xn*yn
            u_dist = fx * (xn + dx_tan) + cx
            v_dist = fy * (yn + dy_tan) + cy
            
        return u, v, u_dist, v_dist

    def plot_stereo_report(self, camera_to_show='left', save_path=None):
        """ 
        Gera o plot comparativo entre Radial e Tangencial da mesma câmera 
        incluindo os dados de calibração abaixo.
        """
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3)
        
        modes = ['radial', 'tangential']
        titles = ['Radial Component of the Distortion Model', 
                  'Tangential Component of the Distortion Model']
        
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

        for ax, mode, title in zip(axes, modes, titles):
            u, v, u_d, v_d = self.compute_field(camera=camera_to_show, mode=mode)
            du, dv = u_d - u, v_d - v
            mags = np.sqrt(du**2 + dv**2)

            # 1. Contornos preenchidos (fundo sutil)
            ax.contourf(u, v, mags, 20, cmap='Greys', alpha=0.15)
            
            # 2. Linhas de contorno com rótulos (Estilo MATLAB)
            levels = 10 if mode == 'radial' else 8
            cs = ax.contour(u, v, mags, levels=levels, colors='black', linewidths=0.6)
            ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')

            # 3. Vetores de distorção (Quiver)
            # Ajuste 'scale' se as setas estiverem muito grandes ou pequenas
            ax.quiver(u, v, du, dv, color='royalblue', alpha=0.8, 
                      scale_units='xy', scale=0.2, width=0.002, headwidth=3)

            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            ax.grid(True, linestyle=':', alpha=0.5)

        # --- Bloco de Texto (Dados da Calibração) ---
        ax_txt = fig.add_subplot(gs[1, :])
        ax_txt.axis('off')
        
        p = self.camera_params[camera_to_show]
        base = np.linalg.norm(self.camera_params['stereo']['T'])
        R = self.camera_params['stereo']['R']
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        yaw = np.degrees(np.arctan2(-R[2,0], sy))
        calib_info = (
            f"CAMERA: {camera_to_show.upper()}\n"
            f"Focal Length: ({p['kk'][0,0]:.2f}, {p['kk'][1,1]:.2f})\n"
            f"Principal Point: ({p['kk'][0,2]:.2f}, {p['kk'][1,2]:.2f})\n"
            f"Radial Coeffs: ({p['kc'][0]:.4f}, {p['kc'][1]:.4f}, {p['kc'][4]:.4f})\n"
            f"Tangential: ({p['kc'][2]:.6f}, {p['kc'][3]:.6f})\n"
            f"Stereo Baseline: {base:.4f} mm\n"
            f"Angle: {yaw:.3f}°"
        )
        
        ax_txt.text(0.5, 0.5, calib_info, family='monospace', fontsize=11,
                    ha='left', va='center_baseline', linespacing=1.5,
                    bbox=dict(boxstyle='round,pad=1', facecolor='#f9f9f9', edgecolor='gray'))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
# Exemplo de uso
# vis = DistortionVisualizer("stereo_calibration.yaml")
# vis.plot_distortion_comparison_like_reference(save_path="advanced_calibration_visual.png")

# Example Usage
# vis = DistortionVisualizer("calibration.yaml")
# vis.plot_distortion(mode='radial', save_path='radial_dist.png')
# vis.plot_distortion(mode='tangential', save_path='tangential_dist.png')

def main():

    calib_uw = "camera_parameters/20260313_uw_P_d2_calib.yaml"  # Path to your calibration YAML file
    calib_lab = "camera_parameters/20260206_binning.yaml"  # Path to your calibration YAML file
    visualizer_uw = DistortionVisualizer(calib_uw, resolution=(1224, 1024))
    visualizer_lab = DistortionVisualizer(calib_lab, resolution=(1224, 1024))
    print("Lab calib distortion coefficients:", visualizer_lab.camera_params['left']['kc'])
    print("UW calib distortion coefficients:", visualizer_uw.camera_params['left']['kc'])

    # Plot and save radial distortion visualization
    visualizer_lab.plot_stereo_report(camera_to_show='left', save_path='left_{}.png'.format(calib_lab.split('/')[-1].split('.')[0]))
    visualizer_lab.plot_stereo_report(camera_to_show='right', save_path='right_{}.png'.format(calib_lab.split('/')[-1].split('.')[0]))
    visualizer_uw.plot_stereo_report(camera_to_show='left', save_path='left_{}.png'.format(calib_uw.split('/')[-1].split('.')[0]))
    visualizer_uw.plot_stereo_report(camera_to_show='right', save_path='right_{}.png'.format(calib_uw.split('/')[-1].split('.')[0]))
if __name__ == "__main__":
    main()