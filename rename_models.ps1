# PowerShell скрипт для переименования NAM моделей
# Упрощает названия, убирая лишнюю информацию

$modelsPath = "assets\nam_models"
$backupPath = "assets\nam_models_backup"

# Создаем backup
Write-Host "Creating backup..." -ForegroundColor Yellow
if (Test-Path $backupPath) {
    Remove-Item $backupPath -Recurse -Force
}
Copy-Item $modelsPath $backupPath -Recurse
Write-Host "Backup created at: $backupPath" -ForegroundColor Green

# Функция для переименования
function Rename-Model {
    param([string]$oldName, [string]$newName)
    
    $oldPath = Join-Path $modelsPath $oldName
    $newPath = Join-Path $modelsPath $newName
    
    if (Test-Path $oldPath) {
        Rename-Item -Path $oldPath -NewName $newPath
        Write-Host "Renamed: $oldName -> $newName" -ForegroundColor Cyan
    }
}

# Маппинг переименований
$renames = @{
    # Peavey 5150
    "Helga B 5150 BlockLetter - Boosted.nam" = "5150 Boosted.nam"
    "Helga B 5150 BlockLetter - NoBoost.nam" = "5150 Clean.nam"
    
    # Peavey 6505+
    "Helga B 6505+ Green ch - MXR Drive.nam" = "6505+ Green M77.nam"
    "Helga B 6505+ Green ch - NoBoost.nam" = "6505+ Green.nam"
    "Helga B 6505+ Red ch - MXR Drive V2.nam" = "6505+ Red M77.nam"
    "Helga B 6505+ Red ch - MXR Drive.nam" = "6505+ Red M77.nam"
    "Helga B 6505+ Red Ch - NoBoost rock.nam" = "6505+ Red Rock.nam"
    "Helga B 6505+ Red ch - NoBoost.nam" = "6505+ Red.nam"
    
    # Peavey 6534+
    "Helga B 6534+ MXR M77 - Helga Behrens.nam" = "6534+ M77.nam"
    "Helga B 6534+ MXR M77.nam" = "6534+ M77.nam"
    "Helga B 6534+ No Boost.nam" = "6534+ Clean.nam"
    "Helga B 6534+ OD808.nam" = "6534+ 808.nam"
    
    # Peavey JSX
    "Helga B JSX Ultra - M77(FIXED).nam" = "JSX Ultra M77.nam"
    "Helga B JSX Ultra - M77(GROUND HUM).nam" = "JSX Ultra M77.nam"
    "Helga B JSX Ultra - No Boost.nam" = "JSX Ultra Clean.nam"
    "Helga B JSX Ultra - OD808.nam" = "JSX Ultra 808.nam"
    "Helga B JSX-Crunch-M77-ESR0,007.nam" = "JSX Crunch M77.nam"
    "Helga B JSX-Crunch-NoBoost-0,004.nam" = "JSX Crunch.nam"
    "Helga B JSX-Crunch-OD808-ESR0,007.nam" = "JSX Crunch 808.nam"
    
    # Peavey XXX
    "Helga B XXX-6L6-M77-ESR0015.nam" = "XXX 6L6 M77.nam"
    "Helga B XXX-6L6-OD808-ESR0015.nam" = "XXX 6L6 808.nam"
    "Helga B XXX-6L6-PP-Rot-MidBoost-ESR056.nam" = "XXX 6L6 Mid Boost.nam"
    "Helga B XXX-6L6-PP-Rot-MidScoop-ESR061.nam" = "XXX 6L6 Mid Scoop.nam"
    "Helga B XXX-6L6-Wendigo-ESR0,016.nam" = "XXX 6L6 Wendigo.nam"
    "Helga B XXX-KT77-M77.nam" = "XXX KT77 M77.nam"
    "Helga B XXX-KT77-NoBoost.nam" = "XXX KT77 Clean.nam"
    "Helga B XXX-KT77-OD808.nam" = "XXX KT77 808.nam"
    
    # Peavey III
    "Helga B III Blue - M77 Boost.nam" = "Peavey III Blue M77.nam"
    "Helga B III Blue - NoBoost.nam" = "Peavey III Blue.nam"
    "Helga B III Red - 805 Boost.nam" = "Peavey III Red 805.nam"
    "Helga B III Red - NoBoost.nam" = "Peavey III Red.nam"
    "Helga B III Red - NoBoostBALLS.nam" = "Peavey III Red Balls.nam"
    
    # Другие Helga B
    "Helga B Boss OS-2.nam" = "Boss OS-2.nam"
    "Helga B GCI Jugendstil.nam" = "GCI Jugendstil.nam"
    "Helga B PP Rot Mid Boost.nam" = "PP Rot Mid Boost.nam"
    "Helga B PP Rot Mid Scoop.nam" = "PP Rot Mid Scoop.nam"
    "Helga B PP Wendigo.nam" = "PP Wendigo.nam"
    
    # Tim R - Peavey 5152
    "Tim R 5152 Clean ish.nam" = "5152 Clean.nam"
    "Tim R 5152 Crunch + TS9.nam" = "5152 Crunch TS9.nam"
    "Tim R 5152 Crunch No Boost.nam" = "5152 Crunch.nam"
    "Tim R 5152 Lead + TS9.nam" = "5152 Lead TS9.nam"
    "Tim R 5152 Lead No Boost.nam" = "5152 Lead.nam"
    
    # Tim R - Marshall JCM2000
    "Tim R JCM2000 Clean.nam" = "JCM2000 Clean.nam"
    "Tim R JCM2000 Crunch 805'd.nam" = "JCM2000 Crunch 805.nam"
    "Tim R JCM2000 Crunch.nam" = "JCM2000 Crunch.nam"
    "Tim R JCM2000 L2 G6 805'd.nam" = "JCM2000 Lead 805.nam"
    "Tim R JCM2000 L2 G6.nam" = "JCM2000 Lead.nam"
    "Tim R JCM2000 L2 G8 805'd.nam" = "JCM2000 Lead 805.nam"
    "Tim R JCM2000 L2 G8.nam" = "JCM2000 Lead.nam"
    
    # Tim R - Marshall JCM900
    "Tim R JCM90050WDualVerbchAG10.nam" = "JCM900 Channel A Gain 10.nam"
    "Tim R JCM90050WDualVerbChAG4.nam" = "JCM900 Channel A Gain 4.nam"
    "Tim R JCM90050WDualVerbCHAG6.nam" = "JCM900 Channel A Gain 6.nam"
    "Tim R JCM90050WDualVerbChAG8.nam" = "JCM900 Channel A Gain 8.nam"
    "Tim R JCM90050WDualVerbChBG12.nam" = "JCM900 Channel B Gain 12.nam"
    "Tim R JCM90050WDualVerbChBG16.nam" = "JCM900 Channel B Gain 16.nam"
    "Tim R JCM90050WDualVerbChBG20.nam" = "JCM900 Channel B Gain 20.nam"
    "Tim R JCM90050WDualVerbChBG8.nam" = "JCM900 Channel B Gain 8.nam"
    
    # Tim R - Jet City
    "Tim R Jet City G2.nam" = "Jet City Gain 2.nam"
    "Tim R Jet City G4.nam" = "Jet City Gain 4.nam"
    "Tim R Jet City G5.nam" = "Jet City Gain 5.nam"
    "Tim R Jet City G6.nam" = "Jet City Gain 6.nam"
    "Tim R Jet City G9 Boosted.nam" = "Jet City Gain 9 Boosted.nam"
    
    # Tim R - Fender
    "Tim R Fender TwinVerb Norm Bright.nam" = "Fender Twin Verb Bright.nam"
    "Tim R Fender TwinVerb Vibrato Bright.nam" = "Fender Twin Verb Vibrato.nam"
    
    # Tim R - Magnatone
    "Tim R Magnatone Super 59 MKii - TS9 On Full.nam" = "Magnatone Super 59 TS9 Full.nam"
    "Tim R Magnatone Super 59 MKii Bridged Blend.nam" = "Magnatone Super 59 Bridged.nam"
    "Tim R Magnatone Super 59 Mkii Bridged BR Bias.nam" = "Magnatone Super 59 Bridged Bias.nam"
    "Tim R Magnatone Super 59 MKii Bridged Driven.nam" = "Magnatone Super 59 Bridged Driven.nam"
    "Tim R Magnatone Super 59 Mkii Bridged Kloned.nam" = "Magnatone Super 59 Bridged Klone.nam"
    "Tim R Magnatone Super 59 Mkii Ch1 Br.nam" = "Magnatone Super 59 Ch1 Bright.nam"
    "Tim R Magnatone Super 59 Mkii Ch1 Norm.nam" = "Magnatone Super 59 Ch1.nam"
    "Tim R Magnatone Super 59 Mkii Ch2 Br.nam" = "Magnatone Super 59 Ch2 Bright.nam"
    "Tim R Magnatone Super 59 Mkii Ch2 Norm.nam" = "Magnatone Super 59 Ch2.nam"
    "Tim R Magnatone Super 59 Mkii Max'd TS9.nam" = "Magnatone Super 59 Max TS9.nam"
    "Tim R Magnatone Super 59 Mkii Max'd.nam" = "Magnatone Super 59 Max.nam"
    "Tim R Magnatone Super 59 Mkii Pushed TS9.nam" = "Magnatone Super 59 Pushed TS9.nam"
    "Tim R Magnatone Super 59 Mkii Pushed.nam" = "Magnatone Super 59 Pushed.nam"
    
    # Tim R - Splawn
    "Tim R Splawn Pro Mod 1st Gear - G12.nam" = "Splawn Pro Mod Gear 1 Gain 12.nam"
    "Tim R Splawn Pro Mod 2nd Gear - G12.nam" = "Splawn Pro Mod Gear 2 Gain 12.nam"
    "Tim R Splawn Pro Mod 3rd Gear - G12.nam" = "Splawn Pro Mod Gear 3 Gain 12.nam"
    "Tim R Splawn Pro Mod G10 Gear 1 OD2.nam" = "Splawn Pro Mod Gear 1 OD2.nam"
    "Tim R Splawn Pro Mod G10 Gear 1.nam" = "Splawn Pro Mod Gear 1.nam"
    "Tim R Splawn Pro Mod G10 Gear 2 OD2.nam" = "Splawn Pro Mod Gear 2 OD2.nam"
    "Tim R Splawn Pro Mod G10 Gear 2.nam" = "Splawn Pro Mod Gear 2.nam"
    "Tim R Splawn Pro Mod G10 Gear 3 OD2.nam" = "Splawn Pro Mod Gear 3 OD2.nam"
    "Tim R Splawn Pro Mod G10 Gear 3.nam" = "Splawn Pro Mod Gear 3.nam"
    "Tim R Splawn Pro Mod G12 Gear 2 OD2 TS9.nam" = "Splawn Pro Mod Gear 2 OD2 TS9.nam"
    "Tim R Splawn Pro Mod G12 Gear 3 OD2 TS9.nam" = "Splawn Pro Mod Gear 3 OD2 TS9.nam"
    
    # Tim R - Педали
    "Tim R Seymour Duncan 805.nam" = "Seymour Duncan 805.nam"
    "Tim R TS9 Driven.nam" = "TS9 Driven.nam"
    "Tim R TS9.nam" = "TS9.nam"
    
    # Keith B - Педали
    "Keith B DS1_g6_t5.nam" = "Boss DS1.nam"
    "Keith B DS1_maxg_t5.nam" = "Boss DS1 Max.nam"
    "Keith B klone_g6_t6_o5.nam" = "Klone.nam"
    "Keith B klone_maxG_t6_o5.nam" = "Klone Max.nam"
    "Keith B Klone_plus_BB_highGain.nam" = "Klone + BB High.nam"
    "Keith B Klone_plus_BB_lowgain.nam" = "Klone + BB Low.nam"
    "Keith B Klone_plus_BB_medGain.nam" = "Klone + BB Medium.nam"
    "Keith B PlumesClone_maxG_t5_switch1.nam" = "Plumes Clone Switch 1.nam"
    "Keith B PlumesClone_maxG_t5_switch3.nam" = "Plumes Clone Switch 3.nam"
    
    # Jason Z
    "Jason Z Boss HM2 v1 kinda bass heavy with medium distortion pure everything turned all the way up tone.nam" = "Boss HM2 Heavy.nam"
    "Jason Z Boss HM2BTFO heavy distortion meant to be standalone less bass more honk.nam" = "Boss HM2 Standalone.nam"
    "Jason Z Boss HM2EQ light distortion with dimed EQ knobs use before an amp.nam" = "Boss HM2 EQ.nam"
    "Jason Z KSR CERES - light blue channel - all noon EQ.nam" = "KSR Ceres Blue.nam"
    "Jason Z KSR CERES - purple channel - all noon EQ.nam" = "KSR Ceres Purple.nam"
    "Jason Z KSR VESTA - light blue channel - all noon EQ.nam" = "KSR Vesta Blue.nam"
    "Jason Z KSR VESTA - purple channel - all noon EQ.nam" = "KSR Vesta Purple.nam"
    "Jason Z Line 6 UBERMETAL INSANE droom metal at its best.nam" = "Line 6 Uber Metal.nam"
    "Jason Z Soldano - Super Lead Overdrive pedal nice crunchy rhythm tone.nam" = "Soldano Super Lead.nam"
    "Jason Z Tech21 dUg DP3X bass preamp pedal all dimed no shift.nam" = "Tech21 dUg DP3X.nam"
    
    # Phillipe P - Bugera
    "Phillipe P Bug1990-Lead-NoDrive-Cab-ESR0,011.nam" = "Bugera 1990 Lead.nam"
    "Phillipe P Bug1990-Lead-NoDrive-ESR0,004.nam" = "Bugera 1990 Lead.nam"
    "Phillipe P Bug333-Clean-Cab-ESR0,007.nam" = "Bugera 333 Clean.nam"
    "Phillipe P Bug333-Clean-ESR0,002.nam" = "Bugera 333 Clean.nam"
    "Phillipe P Bug333-Crunch-DT-Cab-ESR0,014.nam" = "Bugera 333 Crunch DT.nam"
    "Phillipe P Bug333-Crunch-DT-ESR0,0133.nam" = "Bugera 333 Crunch DT.nam"
    "Phillipe P Bug333-Crunch-NoDrive-Cab-ESR0,005.nam" = "Bugera 333 Crunch.nam"
    "Phillipe P Bug333-Crunch-NoDrive-ESR0,002.nam" = "Bugera 333 Crunch.nam"
    "Phillipe P Bug333-Lead-NoDrive-Cab-ESR0,007.nam" = "Bugera 333 Lead.nam"
    "Phillipe P Bug333-Lead-NoDrive-ESR0,004.nam" = "Bugera 333 Lead.nam"
    "Phillipe P Bug6262-Crunch-NoDrive-Cab-ESR0,004.nam" = "Bugera 6262 Crunch.nam"
    "Phillipe P Bugera6262-Crunch-NoDrive-ESR0,002.nam" = "Bugera 6262 Crunch.nam"
    "Phillipe P Bugera6262-Lead-M77-ESR0,020.nam" = "Bugera 6262 Lead M77.nam"
    "Phillipe P Bugera6262-Lead-M77-Gain1,5-ESR0,017.nam" = "Bugera 6262 Lead M77.nam"
    "Phillipe P Bugera6262-Lead-NoDrive-Cab-ESR0,009.nam" = "Bugera 6262 Lead.nam"
    "Phillipe P Bugera6262-Lead-NoDrive-ESR0,009.nam" = "Bugera 6262 Lead.nam"
    
    # Phillipe P - Marshall JVM
    "Phillipe P JVM-CL-GR-NoDrive-Cab-ESR0,007.nam" = "JVM Clean Green.nam"
    "Phillipe P JVM-CL-GR-NoDrive-ESR0,005.nam" = "JVM Clean Green.nam"
    "Phillipe P JVM-CR-OR-NoDrive-ESR0,017.nam" = "JVM Crunch Orange.nam"
    "Phillipe P JVM-CR-OR-SD1-ESR0,018.nam" = "JVM Crunch Orange SD1.nam"
    "Phillipe P JVM-OD1-GR-M77-ESR0,018.nam" = "JVM OD1 Green M77.nam"
    "Phillipe P JVM-OD1-GR-NoDrive-ESR0,015.nam" = "JVM OD1 Green.nam"
    "Phillipe P JVM-OD2-RD-NoDrive-Cab-ESR0,006.nam" = "JVM OD2 Red.nam"
    "Phillipe P JVM-OD2-RD-NoDrive-ESR0,003.nam" = "JVM OD2 Red.nam"
    
    # Phillipe P - Laney
    "Phillipe P LaneyGH100S-Hi-HiNoon-ESR0,024.nam" = "Laney GH100S High.nam"
    "Phillipe P LaneyGH100S-Hi-MidGain-ESR0,003.nam" = "Laney GH100S High Mid.nam"
    "Phillipe P LaneyGH100S-Lo-AfterBreakUp-ESR0,001.nam" = "Laney GH100S Low Breakup.nam"
    "Phillipe P LaneyGH100S-Lo-CrunchGain-ESR0,024.nam" = "Laney GH100S Low Crunch.nam"
    
    # Phillipe P - Педали
    "Phillipe P BOSS-SD1-Feather-ESR0,001.nam" = "Boss SD1.nam"
    "Phillipe P Dirty Tree-Feather-ESR0,001.nam" = "Dirty Tree.nam"
    "Phillipe P Maxon-OD808-Feather-ESR0,001.nam" = "Maxon OD808.nam"
    "Phillipe P MXR-M77-Feather-ESR0,001.nam" = "MXR M77.nam"
    "Phillipe P Precision-Drive-Att4-Feather-ESR0,001.nam" = "Precision Drive Att4.nam"
    "Phillipe P Precision-Drive-Feather-ESR0,001.nam" = "Precision Drive.nam"
    
    # Phillipe P - Vox
    "Phillipe P VOXAC15-JonAr1.nam" = "Vox AC15 JonAr 1.nam"
    "Phillipe P VOXAC15-JonAr2.nam" = "Vox AC15 JonAr 2.nam"
    "Phillipe P VOXAC15-JonAr3.nam" = "Vox AC15 JonAr 3.nam"
    "Phillipe P VOXAC15-JonAr4.nam" = "Vox AC15 JonAr 4.nam"
    "Phillipe P VOXAC15-TopBoost.nam" = "Vox AC15 Top Boost.nam"
    
    # Roman A
    "Roman A BUGERA_333.nam" = "Bugera 333.nam"
    "Roman A BUGERA_333_BOOSTED.nam" = "Bugera 333 Boosted.nam"
    "Roman A LT_MESA_MARKIV_1.nam" = "Mesa Mark IV.nam"
    "Roman A YERASOV_MESHUGGAH.nam" = "Yerasov Meshuggah.nam"
    "Roman A YERASOV_MESHUGGAH_BOOSTED.nam" = "Yerasov Meshuggah Boosted.nam"
    
    # Peter N - Dirty Tree
    "Peter N DirtyTree DT-33_V7_Feather_ESR-0.0001.nam" = "Dirty Tree DT-33 Feather.nam"
    "Peter N DirtyTree DT-33_V7_Lite_ESR-0.0001.nam" = "Dirty Tree DT-33 Lite.nam"
    "Peter N DirtyTree DT-33_V7_Std_ESR-0.0001.nam" = "Dirty Tree DT-33.nam"
    "Peter N DirtyTree DT-TC_V7_L3_H7_Feather-ESR-0.0001.nam" = "Dirty Tree DT-TC Feather.nam"
    "Peter N DirtyTree DT-TC_V7_L3_H7_Lite-ESR-0.0001_(v1).nam" = "Dirty Tree DT-TC Lite.nam"
    "Peter N DirtyTree DT-TC_V7_L3_H7_Std_ESR-0.0001_(v1).nam" = "Dirty Tree DT-TC.nam"
    
    # Peter N - HM2
    "Peter N HM-2_SWEDE_Feather_ESR-0.0097.nam" = "Boss HM2 Swede Feather.nam"
    "Peter N HM-2_SWEDE_Lite_ESR-0.0057.nam" = "Boss HM2 Swede Lite.nam"
    "Peter N HM-2_SWEDE_Std_ESR-0.0034.nam" = "Boss HM2 Swede.nam"
    "Peter N HM-2_V5_L10_H10_D0_Feather_ESR-0.0015.nam" = "Boss HM2 L10 H10 Feather.nam"
    "Peter N HM-2_V5_L10_H10_D0_lite_ESR-0.0007.nam" = "Boss HM2 L10 H10 Lite.nam"
    "Peter N HM-2_V5_L10_H10_D0_Std_ESR-0.0004.nam" = "Boss HM2 L10 H10.nam"
    "Peter N HM-2_V5_L10_H8_D2_Feather-ESR-0.0025.nam" = "Boss HM2 L10 H8 Feather.nam"
    "Peter N HM-2_V5_L10_H8_D2_Lite_ESR-0.0014.nam" = "Boss HM2 L10 H8 Lite.nam"
    "Peter N HM-2_V5_L10_H8_D2_Std_ESR-0.0008.nam" = "Boss HM2 L10 H8.nam"
    "Peter N HM-2_V5_L8_H4_D0_Feather_ESR-0.0025.nam" = "Boss HM2 L8 H4 Feather.nam"
    "Peter N HM-2_V5_L8_H4_D0_Lite_ESR-0.0015.nam" = "Boss HM2 L8 H4 Lite.nam"
    "Peter N HM-2_V5_L8_H4_D0_Std_ESR-0.0015.nam" = "Boss HM2 L8 H4.nam"
    
    # Sascha S
    "Sascha S DirtyShirleyCrunch_G4_BuxBoostTight.nam" = "Dirty Shirley Crunch.nam"
    "Sascha S DirtyShirleyMini_Clean_B1_M6_T7_MV10_G4.nam" = "Dirty Shirley Mini Clean.nam"
    "Sascha S DirtyShirleyMini_crunch_B4_M5.5_T6_MV6_G4_GS_high.nam" = "Dirty Shirley Mini Crunch.nam"
    "Sascha S DirtyShirleyMini_crunch_G6.nam" = "Dirty Shirley Mini Crunch.nam"
    "Sascha S DirtyShirleyMini_crunch_G6_PreQ.nam" = "Dirty Shirley Mini Crunch PreQ.nam"
    "Sascha S DirtyShirleyMini_crunch_G8.nam" = "Dirty Shirley Mini Crunch.nam"
    "Sascha S DirtyShirleyMini_lowDrive_G6_GainStrucLow.nam" = "Dirty Shirley Mini Low Drive.nam"
    "Sascha S DirtyShirleyMini_lowGain_MV10_G3.5_klonboost.nam" = "Dirty Shirley Mini Low Gain.nam"
    "Sascha S FriedmanDSM_PowerAmpEL84_MV10_feather.nam" = "Friedman DSM EL84.nam"
    "Sascha S FriedmanDSM_PowerAmpEL84_MV6_feather.nam" = "Friedman DSM EL84.nam"
    
    # Mikhail K
    "Mikhail K Sovtek MIG50.nam" = "Sovtek MIG50.nam"
    "Mikhail K SovtekMIG50 + DOD FX56B.nam" = "Sovtek MIG50 DOD FX56B.nam"
    "Mikhail K SovtekMIG50 + Klone.nam" = "Sovtek MIG50 Klone.nam"
    "Mikhail K SovtekMIG50 + SparkleDrive.nam" = "Sovtek MIG50 Sparkle Drive.nam"
    "Mikhail K SovtekMIG50 + ThroneTorcher.nam" = "Sovtek MIG50 Throne Torcher.nam"
    
    # Luis R
    "Luis R driftwood purple nightmare TS engaged.nam" = "Driftwood Purple Nightmare TS.nam"
    
    # Tom C
    "Tom C Axe FX 2 Engl Savage.nam" = "Axe FX 2 Engl Savage.nam"
    "Tom C Axe FX 2 Orange Rockerverb.nam" = "Axe FX 2 Orange Rockerverb.nam"
    "Tom C Engl e530 preamp & 840 power amp with Boss DS1 Overdrive.nam" = "Engl E530 840 DS1.nam"
    
    # George B
    "George B Ceriatone King Kong  chan2 60s br sw2 L.nam" = "Ceriatone King Kong Ch2 60s.nam"
    "George B Ceriatone King Kong  KK Ch2 60s solo 80s.nam" = "Ceriatone King Kong Ch2 60s 80s.nam"
    "George B Ceriatone King Kong chan2 70s both br sw L.nam" = "Ceriatone King Kong Ch2 70s.nam"
    "George B Ceriatone King Kong Channel 1 60s mode.nam" = "Ceriatone King Kong Ch1 60s.nam"
    "George B Ceriatone King Kong Channel 2 80s mode both br sw R.nam" = "Ceriatone King Kong Ch2 80s.nam"
    "George B Ceriatone King Kong NAM Capture Bogner Blue Chase Tone Secret Pre Deco.nam" = "Ceriatone King Kong Bogner Blue.nam"
    "George B Ceriatone King Kong NAM Capture Boosted peace keeper Chase Tone Secret Pre Deco v2.nam" = "Ceriatone King Kong Peace Keeper.nam"
    "George B Ceriatone King Kong NAM Capture Broadcast Chase Tone Secret Pre Deco.nam" = "Ceriatone King Kong Broadcast.nam"
    "George B Ceriatone King Kong NAM Capture Protein Chase Tone Secret Pre Deco .nam" = "Ceriatone King Kong Protein.nam"
    "George B Ceriatone King Kong NAM Capture Turbo Nonna Chase Tone Secret Pre Deco.nam" = "Ceriatone King Kong Turbo Nonna.nam"
    "George B V4 Countess 300eps.nam" = "V4 Countess.nam"
    
    # Tudor N - упрощаем только самые важные, остальные слишком специфичные
    "Tudor N Ceriatone Molecular 50.nam" = "Ceriatone Molecular 50.nam"
    "Tudor N Driftwood Nightmare High Gain   HM2.nam" = "Driftwood Nightmare High Gain HM2.nam"
    "Tudor N Driftwood Nightmare Low Gain   TS.nam" = "Driftwood Nightmare Low Gain TS.nam"
    "Tudor N Driftwood Nightmare Low Gain.nam" = "Driftwood Nightmare Low Gain.nam"
}

Write-Host "`nStarting renaming process..." -ForegroundColor Yellow
$count = 0

foreach ($rename in $renames.GetEnumerator()) {
    Rename-Model -oldName $rename.Key -newName $rename.Value
    $count++
}

Write-Host "`nRenaming complete! Processed $count files." -ForegroundColor Green
Write-Host "Backup available at: $backupPath" -ForegroundColor Cyan

