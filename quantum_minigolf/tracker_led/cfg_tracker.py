# =========================
# Config (edit these)
# =========================
CAMERA_INDEX        = 0       # > pick a different camera if you have many; < uses earlier camera indices
FRAME_WIDTH         = 1280    # > more detail, heavier; < faster, less detail
FRAME_HEIGHT        = 720     # > more detail, heavier; < faster, less detail
TARGET_FPS          = 90      # > smoother/more CPU (if camera can deliver); < lighter

THRESHOLD           = 235     # > needs brighter pixels -> fewer detections; < more sensitive -> more noise
MIN_AREA            = 2       # > ignore tiny speckles; < accept small blobs (risk noise)
ASSOC_MAX_PX        = 220.0   # > tolerates faster jumps (risk ID swap); < stricter identity (risk losing track)

# Fixed-size putter (visual only; does NOT scale with LED spacing)
PUTTER_LENGTH_PX    = 380     # > longer drawn putter; < shorter
PUTTER_THICKNESS_PX = 90      # > thicker; < thinner

# Smoothing
VEL_EMA_ALPHA       = 0.45    # > snappier/less lag but noisier speed; < smoother but lags
DIR_EMA_ALPHA       = 0.40    # > quicker orientation; < steadier but lags

# Ball / impact zone
REF_POINT           = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)  # center of the "ball" in px
IMPACT_RADIUS_PX    = 10      # > larger zone -> earlier/easier triggers; < must get closer to trigger
MIN_SPEED_IMPACT    = 100.0   # > require faster swing (fewer false hits); < accept slower moves (more hits)
IMPACT_COOLDOWN_SEC = 0.50    # > fewer duplicate hits; < allows repeated triggers quickly

# Console prints
DEBUG_PRINT_EVERY_N_FRAMES = 0  # >0 prints every N frames; 0 = only print on impact

# Windows (two separate views)
SHOW_LED_WINDOW         = False
SHOW_OVERLAY_WINDOW     = False
ENABLE_OVERLAY_WINDOW   = False
LED_WINDOW_NAME         = "LED Spots (clean)"
OVERLAY_WINDOW_NAME     = "Putter Overlay (clean)"

# LED-only view look
LED_VIEW_MIN_RADIUS     = 6       # minimum drawn radius (px)
LED_VIEW_GLOW           = True    # blur the LED dots for a glow effect
LED_VIEW_GLOW_KERNEL    = 9       # odd number (e.g., 7,9,11). Higher = softer glow

# Overlay HUD
SHOW_OVERLAY_HUD        = True

# Virtual ghost putter overlay (alignment helper)
SHOW_GHOST_PUTTER       = True
GHOST_PUTTER_COLOR      = (80, 220, 255)
GHOST_PUTTER_ALPHA      = 0.4
GHOST_ORIENTATION_DEG   = -90.0   # pointing upward by default
GHOST_CENTER_OFFSET_PX  = (0.0, 0.0)  # offset from REF_POINT

# Impact mode
USE_RECT_IMPACT         = True     # True: detect ball vs putter-rectangle edges; False: legacy center-in-circle

# Region-of-interest tracker acceleration
ROI_SEARCH_MARGIN       = 160     # pixels grown around last LEDs; 0 disables ROI cropping
ROI_MIN_SIZE            = 240     # enforce minimum ROI box (width/height)
ROI_MAX_MISSES          = 4       # after this many misses, fall back to full-frame search

# Small LED fallback tuning
SMALL_LED_THRESHOLD_DELTA = 40   # lower threshold by this amount if initial pass finds nothing
SMALL_LED_MIN_AREA        = 1    # min area when fallback engaged
SMALL_LED_DILATE_ITER     = 1    # dilate iterations to boost sub-pixel LEDs (0 disables)

# Tracking workflow + performance
AUTO_CALIBRATION        = True   # Run step-by-step calibration before tracking loop
USE_GPU_FILTERING       = False   # If CuPy is available, use GPU threshold/blur pipeline


# Overlay refresh decimation (NEW)
OVERLAY_UPDATE_EVERY = 2      # draw overlay every N frames (1 = every frame)#