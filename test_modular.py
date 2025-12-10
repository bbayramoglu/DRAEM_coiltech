import torch
import torch.nn.functional as F
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def load_models(checkpoint_path, base_model_name, gpu_id=0):
    """
    Reconstructive ve Discriminative modelleri yükler ve döndürür.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("Uyarı: GPU bulunamadı, CPU kullanılıyor.")

    reconst_path = os.path.join(checkpoint_path, base_model_name + ".pckl")
    seg_path = os.path.join(checkpoint_path, base_model_name + "_seg.pckl")

    if not os.path.exists(reconst_path):
        raise FileNotFoundError(f"Reconstruction model dosyası bulunamadı: {reconst_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation model dosyası bulunamadı: {seg_path}")

    print(f"Modeller yükleniyor...")
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(reconst_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(seg_path, map_location=device, weights_only=False))
    model_seg.to(device)
    model_seg.eval()
    
    return model, model_seg, device

def prepare_image(image_path, device):
    """
    Görüntüyü okur, yeniden boyutlandırır ve modele girecek tensora çevirir.
    """
    print(f"Görüntü hazırlanıyor: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    
    # Model 256x256 bekliyor
    img_resized = cv2.resize(img, (256, 256))
    
    # Preprocess (0-1 arasına çekme ve tensora çevirme)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_t = np.transpose(img_norm, (2, 0, 1)) # HWC -> CHW
    batch = torch.from_numpy(img_t).unsqueeze(0).to(device) # (1, 3, 256, 256)
    
    return batch, img_resized

def run_inference(model, model_seg, batch):
    """
    Model üzerinde inferans işlemini gerçekleştirir ve sonuçları döndürür.
    Ayrıca inferans süresini de ölçer.
    """
    # GPU senkronizasyonu (daha doğru süre ölçümü için)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        # 1. Görüntüyü yeniden oluştur (Reconstruct)
        gray_rec = model(batch)
        
        # 2. Orijinal ve Reconstruct edilmiş görüntüyü birleştir (Concatenate)
        joined_in = torch.cat((gray_rec.detach(), batch), dim=1)
        
        # 3. Segmentasyon (Anomali haritası)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000 # milisaniye cinsinden
    
    return gray_rec, out_mask_sm, inference_time

def process_results(out_mask_sm):
    """
    Model çıktısından anomali haritasını ve skorunu hesaplar.
    """
    # Anomali haritasını al (Class 1: Anomali)
    anomaly_map = out_mask_sm[0, 1, :, :].cpu().numpy()

    # Skoru hesapla (DRAEM yöntemi: Smoothed Max)
    out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
    image_score = np.max(out_mask_averaged)
    
    return anomaly_map, image_score

def visualize_and_save(img_resized, gray_rec, anomaly_map, image_score, output_path=None):
    """
    Sonuçları görselleştirir ve isteğe bağlı olarak kaydeder.
    """
    # Orijinal Görüntü (Göstermek için BGR -> RGB)
    img_show = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Reconstruct Edilmiş Görüntü (C,H,W -> H,W,C ve BGR -> RGB)
    rec_img = gray_rec[0].permute(1, 2, 0).cpu().numpy()
    rec_img = np.clip(rec_img, 0, 1) # 0-1 arasına sabitle
    rec_img = (rec_img * 255).astype(np.uint8)
    rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)

    # Anomali Haritası Görselleştirme (Heatmap)
    # 0-1 aralığını 0-255'e çek
    heatmap_norm = (anomaly_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Heatmap'i orijinal resmin üzerine bindir
    overlay = cv2.addWeighted(img_show, 0.6, heatmap_color, 0.4, 0)

    # Plot Oluşturma
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    axes[0].imshow(img_show)
    axes[0].set_title("Girdi (Original)")
    axes[0].axis('off')
    
    axes[1].imshow(rec_img)
    axes[1].set_title("Yeniden Oluşturma (Reconstructed)")
    axes[1].axis('off')
    
    # vmin ve vmax değerleri haritanın kontrastını ayarlar
    axes[2].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1) 
    axes[2].set_title("Anomali Haritası (Ham)")
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title(f"Sonuç (Skor: {image_score:.3f})")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Sonuç görseli kaydedildi: {output_path}")
    
    plt.close(fig) # Belleği temizle

def main():
    # --- AYARLAR ---
    CHECKPOINT_PATH = "C:/Users/furkan.bayramoglu/Downloads"
    BASE_MODEL_NAME = "DRAEM_test_0.0001_100_bs16_dr_dataset_"
    GPU_ID = 0
    
    # Test edilecek klasör veya dosya yolu
    TEST_DIR = r"C:\Users\furkan.bayramoglu\Desktop\LAZIM\datasets\18.07_20-23defrom\test\punch"
    OUTPUT_DIR = "./results"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        # 1. Modelleri Yükle
        model, model_seg, device = load_models(CHECKPOINT_PATH, BASE_MODEL_NAME, GPU_ID)
        
        # Klasördeki tüm resimleri işle
        # Eğer sadece tek resim ise burayı düzenleyebilirsiniz
        image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            img_path = os.path.join(TEST_DIR, img_name)
            output_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
            
            # 2. Görüntüyü Hazırla
            batch, img_resized = prepare_image(img_path, device)
            
            # 3. İnferans Yap
            gray_rec, out_mask_sm, inference_time = run_inference(model, model_seg, batch)
            
            # 4. Sonuçları İşle
            anomaly_map, image_score = process_results(out_mask_sm)
            
            print(f"--------------------------------------------------")
            print(f"Dosya: {img_name}")
            print(f"Anomali Skoru: {image_score:.5f}")
            print(f"Süre: {inference_time:.2f} ms")
            print(f"--------------------------------------------------")
            
            # 5. Görselleştir ve Kaydet
            visualize_and_save(img_resized, gray_rec, anomaly_map, image_score, output_path)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()

