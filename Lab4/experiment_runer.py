#!/usr/bin/env python3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def run_cuda_experiments():
    """Запуск CUDA-экспериментов"""
    print("Запуск CUDA экспериментов...")
    subprocess.run(["./matrix_cuda"], check=True)

def analyze_results():
    """Анализ результатов"""
    if not os.path.exists("cuda_results.csv"):
        print("Файл cuda_results.csv не найден. Запустите эксперименты сначала.")
        return
    
    df = pd.read_csv("cuda_results.csv")
    
    plt.figure(figsize=(12, 8))
    
    for block in df['BlockX'].unique():
        for blockY in df[df['BlockX'] == block]['BlockY'].unique():
            mask = (df['BlockX'] == block) & (df['BlockY'] == blockY)
            data = df[mask]
            label = f"{int(block)}x{int(blockY)}"
            plt.plot(data['Size'], data['GFLOPS'], 'o-', label=label, linewidth=2, markersize=8)
    
    plt.xlabel("Matrix Size (N)", fontsize=14)
    plt.ylabel("Performance (GFLOPS)", fontsize=14)
    plt.title("CUDA Matrix Multiplication Performance by Block Configuration", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cuda_performance.png", dpi=150)
    print("Сохранен график: cuda_performance.png")
    
    plt.figure(figsize=(10, 6))
    
    best_by_size = df.loc[df.groupby('Size')['TimeMs'].idxmin()]
    plt.plot(best_by_size['Size'], best_by_size['GFLOPS'], 'ro-', linewidth=2, markersize=10, label='Best CUDA config')
    

    peak_gflops = 3000 
    plt.axhline(y=peak_gflops, color='g', linestyle='--', label=f'Theoretical Peak ({peak_gflops} GFLOPS)')
    
    plt.xlabel("Matrix Size (N)", fontsize=14)
    plt.ylabel("Best Performance (GFLOPS)", fontsize=14)
    plt.title("Best CUDA Performance for Each Matrix Size", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cuda_best_performance.png", dpi=150)
    print("Сохранен график: cuda_best_performance.png")
    
    best_results = []
    for size in best_by_size['Size'].unique():
        row = best_by_size[best_by_size['Size'] == size].iloc[0]
        best_results.append({
            'Size': int(row['Size']),
            'Block Config': f"{int(row['BlockX'])}x{int(row['BlockY'])}",
            'Time (ms)': f"{row['TimeMs']:.2f}",
            'GFLOPS': f"{row['GFLOPS']:.2f}"
        })
    
    best_df = pd.DataFrame(best_results)
    print("\n=== Лучшие конфигурации по размерам ===")
    print(best_df.to_string(index=False))
    
    if os.path.exists("cuda_best_results.csv"):
        best_df_all = pd.read_csv("cuda_best_results.csv")
        print("\n=== Сравнение CPU vs CUDA ===")
        print(best_df_all[['Size', 'BestTimeMs', 'CPUTimeMs', 'Speedup']].to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_df_all['Size'], best_df_all['Speedup'], 'bo-', linewidth=2, markersize=10)
        plt.xlabel("Matrix Size (N)", fontsize=14)
        plt.ylabel("Speedup (CPU Time / CUDA Time)", fontsize=14)
        plt.title("CUDA Speedup over CPU", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("cuda_speedup.png", dpi=150)
        print("Сохранен график: cuda_speedup.png")

if __name__ == "__main__":
    run_cuda_experiments()
    analyze_results()