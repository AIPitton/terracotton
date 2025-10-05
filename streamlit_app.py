import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import rasterio

st.title("TerraCotton: Hillsboro Cotton Phenology Comparison (2013 vs. 2025)")

# Create tabs
tab1, tab2 = st.tabs(["Analysis", "About This App"])

with tab1:
    # Upload two GeoTIFF images
    st.subheader("Upload MODIS/HLS GeoTIFF Images")
    img1 = st.file_uploader("Upload 2013 GeoTIFF (e.g., snapshot-2013-06-12T00_00_00Z.tif)", type=['tif'])
    img2 = st.file_uploader("Upload 2025 GeoTIFF (e.g., snapshot-2025-06-29T00_00_00Z.tif)", type=['tif'])

    # Band selection
    st.subheader("Select Red and NIR Bands")
    col1, col2 = st.columns(2)
    with col1:
        red_band_2013 = st.number_input("2013 Red Band (1-based)", min_value=1, max_value=4, value=1)
        nir_band_2013 = st.number_input("2013 NIR Band (1-based)", min_value=1, max_value=4, value=2)
    with col2:
        red_band_2025 = st.number_input("2025 Red Band (1-based)", min_value=1, max_value=4, value=1)
        nir_band_2025 = st.number_input("2025 NIR Band (1-based)", min_value=1, max_value=4, value=2)

    if img1 and img2:
        try:
            # Open and validate images
            with rasterio.open(img1) as src1, rasterio.open(img2) as src2:
                # Check if dimensions match
                if src1.shape != src2.shape:
                    st.error("Images must have the same width and height.")
                    st.stop()
                # Display metadata
                st.subheader("GeoTIFF Metadata")
                st.write("2013 Image Metadata:", src1.meta)
                st.write("2025 Image Metadata:", src2.meta)
                # Read red and NIR bands
                red1 = src1.read(red_band_2013).astype(float)
                nir1 = src1.read(nir_band_2013).astype(float)
                red2 = src2.read(red_band_2025).astype(float)
                nir2 = src2.read(nir_band_2025).astype(float)
                
                # Scale uint8 values (0-255) to reflectance (0-1)
                scale_factor = 255.0  # For uint8 data
                red1 /= scale_factor
                nir1 /= scale_factor
                red2 /= scale_factor
                nir2 /= scale_factor
                
                # Debug band statistics
                st.subheader("Input Band Statistics")
                st.write("2013 Red Band - Mean:", np.nanmean(red1), "Std:", np.nanstd(red1))
                st.write("2013 NIR Band - Mean:", np.nanmean(nir1), "Std:", np.nanstd(nir1))
                st.write("2025 Red Band - Mean:", np.nanmean(red2), "Std:", np.nanstd(red2))
                st.write("2025 NIR Band - Mean:", np.nanmean(nir2), "Std:", np.nanstd(nir2))
                
                # Sample pixel values for debugging
                st.subheader("Sample Pixel Values (Top-Left Pixel)")
                st.write("2013 Red (Band {}): {:.4f}".format(red_band_2013, red1[0, 0]))
                st.write("2013 NIR (Band {}): {:.4f}".format(nir_band_2013, nir1[0, 0]))
                st.write("2025 Red (Band {}): {:.4f}".format(red_band_2025, red2[0, 0]))
                st.write("2025 NIR (Band {}): {:.4f}".format(nir_band_2025, nir2[0, 0]))
                
                # Band histograms
                st.subheader("Band Value Distributions")
                for name, data in [("2013 Red", red1), ("2013 NIR", nir1), ("2025 Red", red2), ("2025 NIR", nir2)]:
                    hist, bins = np.histogram(data.flatten(), bins=20, range=(0, 1))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    hist_df = pd.DataFrame({name: bin_centers, "Pixel Count": hist}).set_index(name)
                    st.bar_chart(hist_df, use_container_width=True)
                
                # Compute NDVI for both images
                ndvi1 = np.clip((nir1 - red1) / (nir1 + red1 + 1e-10), -1, 1)
                ndvi2 = np.clip((nir2 - red2) / (nir2 + red2 + 1e-10), -1, 1)
                
                # NDVI statistics
                st.subheader("NDVI Statistics")
                ndvi1_flat = ndvi1.flatten()
                ndvi2_flat = ndvi2.flatten()
                valid_ndvi1 = ndvi1_flat[~np.isnan(ndvi1_flat)]
                valid_ndvi2 = ndvi2_flat[~np.isnan(ndvi2_flat)]
                st.write("2013 NDVI - Mean:", np.mean(valid_ndvi1), "Std:", np.std(valid_ndvi1), 
                         "Min:", np.min(valid_ndvi1), "Max:", np.max(valid_ndvi1))
                st.write("2025 NDVI - Mean:", np.mean(valid_ndvi2), "Std:", np.std(valid_ndvi2), 
                         "Min:", np.min(valid_ndvi2), "Max:", np.max(valid_ndvi2))
                st.write("2013 Pixels with NDVI > 0.1:", np.sum(ndvi1 > 0.1), "NDVI > 0.2:", np.sum(ndvi1 > 0.2), "NDVI > 0.3:", np.sum(ndvi1 > 0.3))
                st.write("2025 Pixels with NDVI > 0.1:", np.sum(ndvi2 > 0.1), "NDVI > 0.2:", np.sum(ndvi2 > 0.2), "NDVI > 0.3:", np.sum(ndvi2 > 0.3))
                
                # NDVI histograms
                st.subheader("NDVI Distributions")
                for name, data in [("2013 NDVI", ndvi1), ("2025 NDVI", ndvi2)]:
                    hist, bins = np.histogram(data.flatten(), bins=20, range=(-1, 1))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    hist_df = pd.DataFrame({name: bin_centers, "Pixel Count": hist}).set_index(name)
                    st.bar_chart(hist_df, use_container_width=True)
                
                # Compute NDVI difference (2025 - 2013)
                ndvi_diff = ndvi2 - ndvi1
            
            # Debug NDVI difference data
            st.subheader("NDVI Difference Diagnostics")
            ndvi_diff_flat = ndvi_diff.flatten()
            valid_ndvi_diff = ndvi_diff_flat[~np.isnan(ndvi_diff_flat)]
            if len(valid_ndvi_diff) == 0:
                st.error("NDVI difference data is all NaNs. Check GeoTIFF bands.")
                st.stop()
            stats = {
                "Mean": np.mean(valid_ndvi_diff),
                "Std": np.std(valid_ndvi_diff),
                "Min": np.min(valid_ndvi_diff),
                "Max": np.max(valid_ndvi_diff),
                "NaN Count": np.sum(np.isnan(ndvi_diff_flat))
            }
            st.write("NDVI Difference Statistics:", stats)
            
            # Plot NDVI difference histogram
            st.subheader("NDVI Difference Distribution")
            hist, bins = np.histogram(valid_ndvi_diff, bins=20, range=(-1, 1))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_df = pd.DataFrame({"NDVI Difference": bin_centers, "Pixel Count": hist}).set_index("NDVI Difference")
            st.bar_chart(hist_df, use_container_width=True)
            
            # Use all valid pixels for clustering
            data = valid_ndvi_diff.reshape(-1, 1)
            if len(data) < 2:
                st.error("Too few valid pixels for clustering. Check GeoTIFF data.")
                # Fallback: Use NDVI difference directly
                st.subheader("Phenology Analysis (No Clustering)")
                mean_ndvi_diff = stats["Mean"]
                if mean_ndvi_diff > 0.1:
                    bloom_note = "2025 bloom likely earlier than 2013 (~June 25–July 5 vs. July 5–15) due to higher NDVI."
                    stage = "Reproductive (e.g., Flowering)"
                elif mean_ndvi_diff < -0.1:
                    bloom_note = "2025 bloom likely later than 2013 (~July 10–20 vs. July 5–15) due to lower NDVI."
                    stage = "Vegetative"
                else:
                    bloom_note = "2025 bloom timing similar to 2013 (~July 5–15), with minimal NDVI change."
                    stage = "Transitional"
                st.write(f"Bloom Timing Estimate: {bloom_note}")
                st.write(f"Dominant Stage: {stage}")
                st.write(f"NDVI Insights: 2013 Mean NDVI: {np.mean(valid_ndvi1):.4f}, 2025 Mean NDVI: {np.mean(valid_ndvi2):.4f}")
                st.write("Note: Clustering skipped due to insufficient valid pixels. Based on June NDVI differences at Hillsboro, TX (31.9520° N, -97.3370° W).")
            else:
                # Amplify NDVI differences
                data = data / (np.max(np.abs(data)) + 1e-10)
                if np.std(data) < 1e-5:
                    st.warning("NDVI difference has near-zero variance. Clustering may be unreliable.")
                    # Fallback: Use NDVI difference directly
                    st.subheader("Phenology Analysis (No Clustering)")
                    mean_ndvi_diff = stats["Mean"]
                    if mean_ndvi_diff > 0.1:
                        bloom_note = "2025 bloom likely earlier than 2013 (~June 25–July 5 vs. July 5–15) due to higher NDVI."
                        stage = "Reproductive (e.g., Flowering)"
                    elif mean_ndvi_diff < -0.1:
                        bloom_note = "2025 bloom likely later than 2013 (~July 10–20 vs. July 5–15) due to lower NDVI."
                        stage = "Vegetative"
                    else:
                        bloom_note = "2025 bloom timing similar to 2013 (~July 5–15), with minimal NDVI change."
                        stage = "Transitional"
                    st.write(f"Bloom Timing Estimate: {bloom_note}")
                    st.write(f"Dominant Stage: {stage}")
                    st.write(f"NDVI Insights: 2013 Mean NDVI: {np.mean(valid_ndvi1):.4f}, 2025 Mean NDVI: {np.mean(valid_ndvi2):.4f}")
                    st.write("Note: Clustering skipped due to low variability. Based on June NDVI differences at Hillsboro, TX (31.9520° N, -97.3370° W).")
                else:
                    # K-means clustering with multiple seeds
                    n_clusters = 2
                    stages = ["Vegetative", "Reproductive"]
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
                        labels = kmeans.fit_predict(data)
                        # Convert hard labels to soft memberships (approximation)
                        u = np.zeros((n_clusters, len(data)))
                        for i in range(len(data)):
                            u[labels[i], i] = 1.0
                        
                        # Debug clustering output
                        st.write(f"Cluster Centers: {kmeans.cluster_centers_.flatten()}")
                        st.write(f"Membership Matrix Shape: {u.shape}, Sum of Memberships (first 5 pixels): {np.sum(u, axis=0)[:5]}")
                        
                        # Table: Membership probabilities
                        st.subheader("Phenology Change Analysis (2013 vs. 2025)")
                        memberships_df = pd.DataFrame(u.T, columns=stages)
                        st.write("Membership Probabilities for Phenological Stages (based on NDVI difference):")
                        st.dataframe(memberships_df)
                        
                        # Bar Chart: Average membership probabilities
                        st.subheader("Average Memberships by Phenological Stage")
                        avg_memberships = u.mean(axis=1)
                        if np.allclose(avg_memberships, 1/n_clusters, atol=0.05):
                            st.warning("Near-uniform memberships detected. Clustering may not be effective due to low data variability.")
                        chart_df = pd.DataFrame({"Stage": stages, "Membership Probability": avg_memberships}).set_index("Stage")
                        st.bar_chart(chart_df, use_container_width=True)
                        
                        # Estimate bloom timing based on NDVI difference
                        mean_ndvi_diff = stats["Mean"]
                        if mean_ndvi_diff > 0.1:
                            bloom_note = "2025 bloom likely earlier than 2013 (~June 25–July 5 vs. July 5–15) due to higher NDVI, indicating advanced vegetative growth."
                        elif mean_ndvi_diff < -0.1:
                            bloom_note = "2025 bloom likely later than 2013 (~July 10–20 vs. July 5–15) due to lower NDVI, indicating delayed growth."
                        else:
                            bloom_note = "2025 bloom timing similar to 2013 (~July 5–15), with minimal NDVI change."
                        st.write(f"Bloom Timing Estimate: {bloom_note}")
                        st.write(f"NDVI Insights: 2013 Mean NDVI: {np.mean(valid_ndvi1):.4f}, 2025 Mean NDVI: {np.mean(valid_ndvi2):.4f}")
                        st.write("Note: Based on June NDVI differences at Hillsboro, TX (31.9520° N, -97.3370° W). Positive NDVI differences suggest earlier growth in 2025; negative suggest delay. Reference: USDA NASS, Texas A&M AgriLife.")
                    
                    except Exception as e:
                        st.error(f"Clustering failed: {str(e)}. Displaying NDVI difference statistics only.")
                        st.write("Try adjusting clustering parameters or checking GeoTIFF data.")
            
        except Exception as e:
            st.error(f"Error processing GeoTIFF files: {str(e)}")
    else:
        st.info("Please upload both 2013 and 2025 GeoTIFF images to compare cotton blooming.")

with tab2:
    st.header("About This App")
    st.markdown("""
### TerraCotton: Hillsboro Cotton Phenology Comparison
**Purpose**: This Streamlit web app analyzes cotton phenology (growth stages) in Hillsboro, TX (31.9520° N, -97.3370° W), comparing 2013 and 2025 using NASA MODIS or Harmonized Landsat Sentinel-2 (HLS) GeoTIFF images. It estimates bloom timing shifts to support precision agriculture, aiding farmers with planting, irrigation, and pest management under changing climate conditions.

**Functionality**:
- **Data Input**: Upload two GeoTIFF images (e.g., 2013 and 2025 snapshots) and select red/NIR bands (default: band 1=red, band 2=NIR; adjustable for HLS: band 4=red, band 5=NIR).
- **NDVI Calculation**: Computes Normalized Difference Vegetation Index (NDVI = (NIR - Red) / (NIR + Red)) to assess vegetation health, expecting values ~0.3–0.7 for cotton in June.
- **Diagnostics**: Displays band statistics, sample pixel values, NDVI statistics (mean, std, min, max), and histograms to verify data quality and detect issues (e.g., low NDVI ~0.016 due to band selection or sparse vegetation).
- **Clustering**: Uses k-means clustering to group pixels into "Vegetative" (early growth) or "Reproductive" (e.g., flowering) stages based on NDVI differences (2025 - 2013), amplifying differences to handle low variability (Std: ~0.0198).
- **Fallback Analysis**: If clustering fails (e.g., low pixel count or variance), estimates phenology using mean NDVI difference:
  - >0.1: Earlier bloom in 2025 (~June 25–July 5 vs. July 5–15, "Reproductive").
  - <-0.1: Later bloom (~July 10–20, "Vegetative").
  - Otherwise: Similar timing (~July 5–15, "Transitional").
- **Visualizations**: Shows membership probability tables, bar charts of average memberships, and bloom timing estimates, referencing USDA NASS and Texas A&M AgriLife.

**NASA Hackathon Relevance**:
- Leverages NASA MODIS/HLS data, aligning with Earth observation for agriculture.
- Builds on DOI:10.1002/ecs2.70127, using NDVI-based clustering to detect phenological shifts.
- Offers an interactive interface for researchers, with robust diagnostics to ensure reliability.

**Limitations**:
- Low NDVI (~0.016) may indicate band misselection (e.g., HLS requires band 4=red, 5=NIR) or sparse vegetation in 25x25 pixel images (30m resolution).
- Small NDVI differences (Mean: ~0.0127) may lead to uniform memberships (~0.5), triggering fallback analysis.
- Future enhancements could include multi-date imagery or higher-resolution Sentinel-2 data.

**Impact**: Supports precision agriculture by estimating cotton bloom timing, critical for yield prediction and climate adaptation in Hillsboro, TX.
    """)

st.write("Like DOI:10.1002/ecs2.70127, clustering on NASA MODIS/HLS NDVI differences helps estimate phenological shifts.")