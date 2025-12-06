import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.aperture import CircularAperture 
from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats, aperture_photometry
import math

r = 8.0  # радиус апертуры
fwhm = 4.305  # FWHM в пикселях
date = '2016-04-04'
instrument = 'WFC'


zeropoints = {
    'B': 25.652,  
    'V': 25.0,    
    'R': 24.5     
}

root_dir = "C:/Users/nazym/OneDrive/Рабочий стол/Physics NIS/FAI/NGC4151/B_2016-04-04"

# Список для хранения путей ко всем FITS файлам
all_fit_files = {"B": [], "V": [], "R": []}

# Сканируем папки B, V, R 
for band in ["B", "V", "R"]:
    band_dir = os.path.join(root_dir, band)
    if os.path.exists(band_dir):  # проверяем существование папки
        for path, subdirs, files in os.walk(band_dir):
            for name in files:
                if name.lower().endswith((".fit", ".fits")):  
                    full_path = os.path.join(path, name)
                    all_fit_files[band].append(full_path)
        print(f"Найдено {len(all_fit_files[band])} файлов в папке {band}")
    else:
        print(f"⚠ Папка {band_dir} не существует")

all_results = {}


target_x = 800
target_y = 834
tolerance = 5.0  

# Чтение и обработка каждого FITS файла в каждой из папок
for band, fit_files in all_fit_files.items():
    print(f"\nОбрабатываю папку: {band}, файлов: {len(fit_files)}")
    
    if len(fit_files) == 0:
        print(f" Нет файлов в папке {band}, пропускаю")
        continue

    for file_path in fit_files:
        print(f"Открываю файл: {os.path.basename(file_path)}")

        try:
            # Открытие FITS файла
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data.astype(float)
                header = hdul[0].header
            
            # Получаем время экспозиции из заголовка
            possible_exptime_keys = ['EXPTIME', 'EXP_TIME', 'TEXPTIME', 'ELAPSED']
            exptime = 90.0

            for key in possible_exptime_keys:
                if key in header:
                    exptime = header[key]
                    print(f"Найден ключ {key}: {exptime} сек")
                    break
            else:
                print(" Не найден ни один из ключей времени экспозиции, использую 90.0 сек по умолчанию")
            
            date_obs = header.get("DATE-OBS", None)
            if date_obs is None:
                date_obs = header.get("DATE-OBS", header.get("DATE-OBS", None))

            # Конвертируем DATE-OBS → JD
            jd = None
            if date_obs is not None:
                try:
                    t = Time(date_obs, format='isot', scale='utc')
                    jd = float(t.jd)
                except:
                    try:
                        t = Time(date_obs, scale='utc')
                        jd = float(t.jd)
                    except:
                        print(f" Не удалось обработать дату: {date_obs}")
                        continue
            else:
                print(f" DATE-OBS отсутствует в файле")
                continue

            # Статистика изображения
            mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
            
            # Поиск звезд
            daofind = DAOStarFinder(fwhm=fwhm, threshold=5.0 * std)  
            sources = daofind(image_data - median)
            
            if sources is None:
                print(f" Не найдено звезд в файле {os.path.basename(file_path)}")
                continue
                
            # Создаем позиции для апертур
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

            # Создаем апертуры
            apertures = CircularAperture(positions, r=r)
            annulus_aperture = CircularAnnulus(positions, r_in=38, r_out=50)

            # Вычисляем фон
            aperstats = ApertureStats(image_data, annulus_aperture)
            bkg_mean = aperstats.mean
            aperture_area = apertures.area_overlap(image_data)
            total_bkg = bkg_mean * aperture_area

            # Фотометрия
            star_data = aperture_photometry(image_data, apertures)
            star_data['total_bkg'] = total_bkg
            star_data['aperture_sum_bkg_subtracted'] = star_data['aperture_sum'] - total_bkg

            # Ищем целевую звезду
            found_target = False
            target_magnitude = None
            
            for i, source in enumerate(sources):
                x_centroid = source['xcentroid']
                y_centroid = source['ycentroid']
                
                distance = np.sqrt((x_centroid - target_x)**2 + (y_centroid - target_y)**2)
                
                if distance < tolerance:
                    print(f" Найдена целевая звезда на расстоянии {distance:.2f} пикселей")
                    
                    # Получаем соответствующие данные фотометрии
                    aperture_sum_bkg_subtracted = star_data['aperture_sum_bkg_subtracted'][i]
                    
                    # Вычисляем звездную величину
                    if aperture_sum_bkg_subtracted > 0:
                        target_magnitude = zeropoints[band] - 2.5 * math.log10(aperture_sum_bkg_subtracted / exptime)
                        found_target = True
                        break

            if found_target and target_magnitude is not None:
                # Сохраняем результаты
                if jd not in all_results:
                    all_results[jd] = {
                        "JD": jd,
                        "Date": date_obs,
                        "Mag_B": None,
                        "Mag_V": None,
                        "Mag_R": None
                    }
                
                # Сохраняем величину в соответствующий фильтр
                if band == "B":
                    all_results[jd]["Mag_B"] = target_magnitude
                elif band == "V":
                    all_results[jd]["Mag_V"] = target_magnitude
                elif band == "R":
                    all_results[jd]["Mag_R"] = target_magnitude
                    
                print(f" Величина в фильтре {band}: {target_magnitude:.3f}")
            else:
                print(f" Целевая звезда не найдена в файле {os.path.basename(file_path)}")

        except Exception as e:
            print(f" Ошибка при обработке файла {file_path}: {str(e)}")
            continue

# Сохранение результатов
if all_results:
    df = pd.DataFrame(list(all_results.values()))
    df = df.sort_values(by="JD")

    output_file = "C:/Users/nazym/OneDrive/Рабочий стол/Physics NIS/FAI/NGC4151/B_2016-04-04/photometry_results.xlsx"
    df.to_excel(output_file, index=False)

    print(f"\n Готово! Результаты сохранены в: {output_file}")
    print(f"Обработано {len(all_results)} измерений")
else:
    print(" Не удалось получить ни одного результата")