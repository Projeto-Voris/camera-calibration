import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

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
        Plots left and right camera distortion side-by-side with stereo metadata.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        sides = ['left', 'right']
        
        for i, side in enumerate(sides):
            u, v, u_dist, v_dist = self.compute_distortion_field(camera=side, mode=mode)
            
            # Quiver plot for distortion vectors
            # scale=0.1 magnifies the visual effect for better clarity in articles
            ax = axes[i]
            ax.quiver(u, v, u_dist - u, v_dist - v, 
                      scale_units='xy', scale=0.1, color='teal', alpha=0.7)
            
            ax.set_title(f"{side.upper()} Camera - {mode.capitalize()} Distortion", fontsize=14)
            ax.invert_yaxis()
            ax.set_xlabel("Pixels (X)")
            if i == 0: ax.set_ylabel("Pixels (Y)")
            ax.grid(True, linestyle=':', alpha=0.4)

        # --- Calculate Stereo Metadata ---
        T = self.camera_params['stereo']['T']
        R = self.camera_params['stereo']['R']
        
        # Baseline is the Euclidean distance between camera centers
        baseline = np.linalg.norm(T)
        
        # Convert Rotation Matrix to Euler Angles (Degrees) for readability
        # Assuming typical OpenCV coordinate system (Z forward, X right, Y down)
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        euler_deg = np.degrees([x, y, z])

        # Add metadata text box
        stats_text = (
            f"Stereo Baseline: {baseline:.4f} units\n"
            f"Relative Rotation (XYZ Deg):\n"
            f"[{euler_deg[0]:.2f}°, {euler_deg[1]:.2f}°, {euler_deg[2]:.2f}°]"
        )
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Publication-ready plot saved to: {save_path}")
            
        plt.show()

# Example Usage
# vis = DistortionVisualizer("calibration.yaml")
# vis.plot_distortion(mode='radial', save_path='radial_dist.png')
# vis.plot_distortion(mode='tangential', save_path='tangential_dist.png')

def main():

    calib_uw = "camera_parameters/20260313_uw_P_d2_calib.yaml"  # Path to your calibration YAML file
    calib_lab = "camera_parameters/20260206_binning.yaml"  # Path to your calibration YAML file
    visualizer_uw = DistortionVisualizer(calib_uw)
    visualizer_lab = DistortionVisualizer(calib_lab)
    print("Lab calib distortion coefficients:", visualizer_lab.camera_params['left']['kc'])
    print("UW calib distortion coefficients:", visualizer_uw.camera_params['left']['kc'])

    # Plot and save radial distortion visualization
    visualizer_uw.plot_distortion(mode='radial', save_path='radial_{}.png'.format(calib_uw.split('/')[-1].split('.')[0]))
    visualizer_uw.plot_distortion(mode='tangential', save_path='tangential_{}.png'.format(calib_uw.split('/')[-1].split('.')[0]))
    # visualizer_lab.plot_distortion(mode='radial', save_path='radial_{}.png'.format(calib_lab.split('/')[-1].split('.')[0]))
    # visualizer_lab.plot_distortion(mode='tangential', save_path='tangential_{}.png'.format(calib_lab.split('/')[-1].split('.')[0]))

    # Plot and save tangential distortion visualization

if __name__ == "__main__":
    main()