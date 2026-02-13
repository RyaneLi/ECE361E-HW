"""
Generate HW2 Solution PDF
Compiles all answers, plots, tables, and open-ended responses into a single PDF document.
"""
import pandas as pd
from fpdf import FPDF
import os
from PIL import Image
import io

class HW2_PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'ECE361E - Homework 2 Solutions', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
        
    def body_text(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(2)
        
    def add_image_full_width(self, image_path, title=""):
        if os.path.exists(image_path):
            if title:
                self.section_title(title)
            # Calculate image width to fit page
            page_width = self.w - 2 * self.l_margin
            
            # Convert PNG to RGB JPG to avoid fpdf PNG parsing issues
            temp_path = image_path
            if image_path.lower().endswith('.png'):
                try:
                    img = Image.open(image_path)
                    # Convert RGBA to RGB if needed
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as temporary JPG
                    temp_path = image_path.rsplit('.', 1)[0] + '_temp.jpg'
                    img.save(temp_path, 'JPEG', quality=95)
                except Exception as e:
                    self.body_text(f"[Error processing image {image_path}: {str(e)}]")
                    return
            
            try:
                self.image(temp_path, x=self.l_margin, w=page_width)
                # Clean up temp file
                if temp_path != image_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                self.body_text(f"[Error adding image {image_path}: {str(e)}]")
                if temp_path != image_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            
            self.ln(5)
        else:
            self.body_text(f"[Image not found: {image_path}]")
            
    def add_csv_table(self, csv_path, title=""):
        if title:
            self.section_title(title)
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            
            # Table styling
            self.set_font('Arial', 'B', 9)
            self.set_fill_color(230, 230, 230)
            
            # Calculate column widths
            num_cols = len(df.columns) + 1  # +1 for index
            col_width = (self.w - 2 * self.l_margin) / num_cols
            
            # Header row
            self.cell(col_width, 7, df.index.name or '', 1, 0, 'C', 1)
            for col in df.columns:
                self.cell(col_width, 7, str(col), 1, 0, 'C', 1)
            self.ln()
            
            # Data rows
            self.set_font('Arial', '', 9)
            for idx, row in df.iterrows():
                self.cell(col_width, 6, str(idx), 1, 0, 'L')
                for val in row:
                    self.cell(col_width, 6, f'{val:.4f}' if isinstance(val, float) else str(val), 1, 0, 'C')
                self.ln()
            
            self.ln(5)
        else:
            self.body_text(f"[Table not found: {csv_path}]")


def generate_hw2_pdf():
    """Generate the complete HW2 solution PDF"""
    
    pdf = HW2_PDF()
    pdf.add_page()
    
    # Title page
    pdf.set_font('Arial', 'B', 20)
    pdf.ln(30)
    pdf.cell(0, 10, 'ECE361E', 0, 1, 'C')
    pdf.cell(0, 10, 'Homework 2 - Complete Solutions', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Edge Computing Systems', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Names: Resul Ovezov, Ryane Li', 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, 'Resul: Completed Problems 1 and 2', 0, 1, 'C')
    pdf.cell(0, 6, 'Ryane: Completed Problem 3', 0, 1, 'C')
    
    # ========================================================================
    # PROBLEM 1: Odroid MC1 Benchmark Measurements
    # ========================================================================
    pdf.add_page()
    pdf.chapter_title('Problem 1: Odroid MC1 Benchmark Measurements')
    
    # Question 1
    pdf.section_title('Question 1: TPBench Measurement [12p]')
    
    pdf.add_image_full_width('p1_q1_plot1_power.png', 'Plot 1: System Power Consumption')
    pdf.add_page()
    pdf.add_image_full_width('p1_q1_plot2_temperature.png', 'Plot 2: Big Cores Temperature')
    pdf.add_page()
    pdf.add_image_full_width('p1_q1_plot3_usage.png', 'Plot 3: Big Cores Usage')
    
    # Question 2
    pdf.add_page()
    pdf.section_title('Question 2: Benchmark Phase Identification [3p]')
    pdf.body_text('Question: How many phases of benchmark execution can you identify based on the temperature variation in the plot? A phase is a significant increase in the temperature over an extended period of time.')
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 11)
    pdf.multi_cell(0, 6, 'Answer:')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 'Based on the temperature variation in the TPBench temperature plot, 4 distinct phases can be identified.')
    pdf.ln(5)
    
    # Question 3
    pdf.section_title('Question 3: Blackscholes and Bodytrack Benchmarks [15p]')
    
    pdf.add_image_full_width('p1_q3_blackscholes_plots.png', 'Blackscholes Benchmark Plots')
    pdf.add_page()
    pdf.add_image_full_width('p1_q3_bodytrack_plots.png', 'Bodytrack Benchmark Plots')
    pdf.add_page()
    pdf.add_csv_table('p1_q3_table1_metrics.csv', 'Table 1: Benchmark Metrics')
    
    # ========================================================================
    # PROBLEM 2: System Power Prediction
    # ========================================================================
    pdf.add_page()
    pdf.chapter_title('Problem 2: System Power Prediction')
    
    # Question 1
    pdf.section_title('Question 1: SVM Classification [20p]')
    
    pdf.add_csv_table('p2_q1_table2_metrics.csv', 'Table 2: SVM Classification Performance')
    pdf.add_image_full_width('p2_q1_confusion_matrices.png', 'Confusion Matrices')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.multi_cell(0, 6, 'Performance Analysis:')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, 'The performance is terrible. The model just ends up predicting everything as active and does nothing else on the testing set even though it is accurate on the training set and is able to differentiate. Features were normalized but still the performance is terrible.')
    pdf.ln(5)
    
    # Question 2
    pdf.section_title('Question 2: Linear Regression for Power Prediction [5p]')
    
    pdf.add_csv_table('p2_q2_table3_metrics.csv', 'Table 3: Linear Regression Performance')
    pdf.add_page()
    pdf.add_image_full_width('p2_q2_power_prediction.png', 'Power Prediction Results')
    
    # Question 3
    pdf.add_page()
    pdf.section_title('Question 3: Feature Engineering with Vdd^2*f [15p]')
    
    pdf.add_image_full_width('p2_q3_feature_importance.png', 'Feature Importance Analysis')
    
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.multi_cell(0, 6, 'Top 3 Positive Features Contributing to Big Cluster Power:')
    pdf.set_font('Arial', '', 11)
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 6, '1. vdd2_f (Vdd^2*f) [1.514170]:')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'This feature based on the dynamic power formula P proportional to V^2*f has the strongest positive correlation with power consumption, confirming the theoretical relationship between voltage, frequency, and dynamic power.')
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 6, '2. temp4 [1.001162]:')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'Temperature of Core 4. Most likely the core most frequently assigned by the OS to take on tasks. Assuming that is the case, more tasks lead to more power consumption which generates more heat.')
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 6, '3. temp5 [0.681568]:')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'Temperature of Core 5. Most likely the core second most frequently assigned by the OS to take on tasks, although less by a decent margin.')
    pdf.ln(3)
    
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 6, 'Note: The feature importance might be different depending on how parallelizable the task is that\'s running.')
    
    # ========================================================================
    # PROBLEM 3: Temperature Prediction
    # ========================================================================
    pdf.add_page()
    pdf.chapter_title('Problem 3: Temperature Prediction using Neural Networks')
    
    # Question 1
    pdf.section_title('Question 1: MLPRegressor for Temperature Prediction')
    
    pdf.add_image_full_width('blackscholes_core4_prediction.png', 'Blackscholes - Core 4 Temperature Prediction')
    pdf.add_page()
    pdf.add_image_full_width('bodytrack_core4_prediction.png', 'Bodytrack - Core 4 Temperature Prediction')
    
    # Question 2
    pdf.add_page()
    pdf.section_title('Question 2: Techniques to Improve Regressor Performance')
    pdf.body_text('Two or more techniques that can improve the temperature prediction model:')
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 6, '1. Feature scaling/normalization:')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'Apply StandardScaler or MinMaxScaler to non-temperature features (e.g., usage_c4-usage_c7, freq_big_cluster). Scaling puts predictors on a similar scale so the MLP converges faster and often generalizes better.')
    pdf.ln(2)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.multi_cell(0, 6, '2. Temporal features:')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, 'Add lagged temperatures (e.g., from t-2, t-3) or rolling statistics (moving average, standard deviation) as features. This helps capture dynamics and thermal inertia.')
    
    # Save PDF
    output_path = 'HW2_Complete_Solutions.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    # Change to the directory containing all the files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Generating HW2 Complete Solutions PDF...")
    print("=" * 70)
    
    output_file = generate_hw2_pdf()
    
    print("=" * 70)
    print(f"✓ SUCCESS: PDF generated at {output_file}")
    print("\nThe PDF includes:")
    print("  - Problem 1: All 3 questions with plots and answers")
    print("  - Problem 2: All 3 questions with tables, plots, and analysis")
    print("  - Problem 3: Both questions with plots and improvement suggestions")
