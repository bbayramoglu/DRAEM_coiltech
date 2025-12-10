import torch
import torch.nn.functional as F
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

def test_single_image(image_path, checkpoint_path, base_model_name, output_path, gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("Uyarı: GPU bulunamadı, CPU kullanılıyor.")

    # Model dosya yolları
    # Not: test_DRAEM.py'de model adı şöyle oluşturuluyor: base_model_name + "_" + obj_name + "_"
    # Bu scriptte doğrudan tam model adını (uzantısız) 'base_model_name' olarak vermelisiniz.
    # Örn: DRAEM_test_0.0001_20_bs16_18.07_20-23defrom_
    
    reconst_path = os.path.join(checkpoint_path, base_model_name + ".pckl")
    seg_path = os.path.join(checkpoint_path, base_model_name + "_seg.pckl")

    if not os.path.exists(reconst_path):
        print(f"HATA: Reconstruction model dosyası bulunamadı: {reconst_path}")
        return
    if not os.path.exists(seg_path):
        print(f"HATA: Segmentation model dosyası bulunamadı: {seg_path}")
        return

    # --- Modelleri Yükle ---
    print(f"Modeller yükleniyor...")
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(reconst_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(seg_path, map_location=device, weights_only=False))
    model_seg.to(device)
    model_seg.eval()

    # --- Görüntüyü Hazırla ---
    print(f"Görüntü işleniyor: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"HATA: Görüntü okunamadı.")
        return
    
    # Model 256x256 bekliyor
    img_resized = cv2.resize(img, (256, 256))
    
    # Preprocess (0-1 arasına çekme ve tensora çevirme)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_t = np.transpose(img_norm, (2, 0, 1)) # HWC -> CHW
    batch = torch.from_numpy(img_t).unsqueeze(0).to(device) # (1, 3, 256, 256)

    # --- İnferans (Tahmin) ---
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

    # --- Sonuçları İşle ---
    # Anomali haritasını al (Class 1: Anomali)
    anomaly_map = out_mask_sm[0, 1, :, :].cpu().numpy()

    # Skoru hesapla (DRAEM yöntemi: Smoothed Max)
    out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
    image_score = np.max(out_mask_averaged)
    
    print(f"--------------------------------------------------")
    print(f"Görüntü Anomali Skoru: {image_score:.5f}")
    print(f"İnferans Süresi: {inference_time:.2f} ms")
    print(f"--------------------------------------------------")

    # --- Görselleştirme ---
    # Orijinal Görüntü (Göstermek için BGR -> RGB)
    img_show = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Reconstruct Edilmiş Görüntü (C,H,W -> H,W,C ve BGR -> RGB)
    rec_img = gray_rec[0].permute(1, 2, 0).cpu().numpy()
    rec_img = np.clip(rec_img, 0, 1) # 0-1 arasına sabitle
    rec_img = (rec_img * 255).astype(np.uint8)
    # Model çıktısı da BGR formatında eğitildiği varsayılıyor (OpenCV okumasıyla), o yüzden RGB'ye çeviriyoruz
    # Eğer renkler ters çıkarsa buradaki dönüşümü kaldırın.
    rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)

    # Anomali Haritası Görselleştirme (Heatmap)
    # 0-1 aralığını 0-255'e çek
    heatmap_norm = (anomaly_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    # Heatmap OpenCV BGR döner, Matplotlib RGB bekler
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
    
    axes[2].imshow(anomaly_map, cmap='jet', vmin=0.1, vmax=0.7)
    axes[2].set_title("Anomali Haritası (Ham)")
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title(f"Sonuç (Skor: {image_score:.3f})")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Sonuç görseli kaydedildi: {output_path}")
    
    # Eğer interaktif ortamdaysa göster
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRAEM Tek Resim Testi")
    parser.add_argument('--image_path', required=True, type=str, help="Test edilecek resmin tam yolu")
    parser.add_argument('--base_model_name', required=True, type=str, help="Model dosya adı (uzantısız). Örn: 'DRAEM_test_..._obj_'")
    parser.add_argument('--checkpoint_path', required=True, type=str, help="Model dosyalarının bulunduğu klasör yolu")
    parser.add_argument('--output_path', type=str, default="single_test_result.png", help="Sonuç görselinin kaydedileceği dosya adı")
    parser.add_argument('--gpu_id', type=int, default=0, help="Kullanılacak GPU ID (Varsayılan: 0)")
    
    args = parser.parse_args()

    
    test_single_image(args.image_path, args.checkpoint_path, args.base_model_name, args.output_path, args.gpu_id)


