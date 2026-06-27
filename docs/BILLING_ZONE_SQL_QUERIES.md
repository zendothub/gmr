# Billing Zone & Rule SQL Queries

## Complete SQL Guide for Setting Up Purchase Tracking

This guide provides SQL queries to manually create billing zones and rules for purchase tracking.

---

## Prerequisites

1. **Get your camera ID:**
```sql
SELECT id, name, rtsp_url, is_active 
FROM cameras 
WHERE is_active = true 
ORDER BY created_at DESC;
```

Save the camera ID (UUID) for the camera you want to track purchases on.

---

## Step 1: Create Billing Zone

### Method A: Using Percentage-Based Coordinates (Recommended)

This method uses percentage coordinates (0-100%) which work with any camera resolution:

```sql
-- Replace YOUR_CAMERA_ID with your actual camera UUID
INSERT INTO zones (
    id,
    camera_id,
    name,
    zone_type,
    shape,
    polygon,
    is_active,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid(),                    -- Auto-generate zone ID
    'YOUR_CAMERA_ID',                     -- Replace with your camera UUID
    'Billing Counter',                    -- Zone name
    'billing_zone',                       -- Must be 'billing_zone'
    'polygon',                            -- Shape type
    '{"points": [[30, 30], [70, 30], [70, 70], [30, 70]]}'::jsonb,  -- Center area (30-70% x, 30-70% y)
    true,                                 -- Zone is active
    NOW(),
    NOW()
)
RETURNING id, name, zone_type;
```

**Common Polygon Configurations:**

```sql
-- Full frame coverage (if counter takes most of the view)
'{"points": [[10, 10], [90, 10], [90, 90], [10, 90]]}'::jsonb

-- Bottom half (counter at bottom of frame)
'{"points": [[0, 50], [100, 50], [100, 100], [0, 100]]}'::jsonb

-- Right side (counter on right)
'{"points": [[60, 0], [100, 0], [100, 100], [60, 100]]}'::jsonb

-- Center area (default - safest option)
'{"points": [[30, 30], [70, 30], [70, 70], [30, 70]]}'::jsonb
```

### Verify Zone Creation:

```sql
SELECT 
    z.id,
    z.name,
    z.zone_type,
    z.shape,
    z.polygon,
    z.is_active,
    c.name as camera_name
FROM zones z
JOIN cameras c ON c.id = z.camera_id
WHERE z.zone_type = 'billing_zone'
ORDER BY z.created_at DESC;
```

---

## Step 2: Create Billing Interaction Rule

```sql
-- Get the zone ID from Step 1
-- Replace YOUR_ZONE_ID and YOUR_CAMERA_ID with actual UUIDs

INSERT INTO rules (
    id,
    camera_id,
    zone_id,
    name,
    rule_type,
    config,
    cooldown_seconds,
    severity,
    dwell_threshold_seconds,
    is_enabled,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid(),                    -- Auto-generate rule ID
    'YOUR_CAMERA_ID',                     -- Same camera as zone
    'YOUR_ZONE_ID',                       -- Zone ID from Step 1
    'Billing Counter Interaction',        -- Rule name
    'billing_interaction',                -- Must be 'billing_interaction'
    '{"threshold_seconds": 5}'::jsonb,    -- Person must stay 5 seconds
    30,                                   -- Cooldown: 30 seconds between triggers
    'info',                               -- Severity level
    5,                                    -- Dwell threshold: 5 seconds
    true,                                 -- Rule is enabled
    NOW(),
    NOW()
)
RETURNING id, name, rule_type, dwell_threshold_seconds;
```

### Verify Rule Creation:

```sql
SELECT 
    r.id,
    r.name,
    r.rule_type,
    r.dwell_threshold_seconds,
    r.cooldown_seconds,
    r.is_enabled,
    z.name as zone_name,
    c.name as camera_name
FROM rules r
JOIN zones z ON z.id = r.zone_id
JOIN cameras c ON c.id = r.camera_id
WHERE r.rule_type = 'billing_interaction'
ORDER BY r.created_at DESC;
```

---

## Step 3: Complete Setup Verification

```sql
SELECT 
    c.id as camera_id,
    c.name as camera_name,
    c.is_active as camera_active,
    z.id as zone_id,
    z.name as zone_name,
    z.zone_type,
    z.is_active as zone_active,
    r.id as rule_id,
    r.name as rule_name,
    r.rule_type,
    r.dwell_threshold_seconds,
    r.is_enabled as rule_enabled
FROM cameras c
LEFT JOIN zones z ON z.camera_id = c.id AND z.zone_type = 'billing_zone'
LEFT JOIN rules r ON r.zone_id = z.id AND r.rule_type = 'billing_interaction'
WHERE c.is_active = true
ORDER BY c.created_at DESC;
```

**Expected Output:**
- Camera should be `camera_active = true`
- Zone should exist with `zone_active = true`
- Rule should exist with `rule_enabled = true`
- All IDs should be filled (not NULL)

---

## Step 4: Monitor Billing Interactions

### Check if interactions are being created:

```sql
SELECT 
    bi.id,
    bi.entered_at,
    bi.exited_at,
    bi.dwell_seconds,
    c.name as camera_name,
    z.name as zone_name
FROM billing_interactions bi
JOIN cameras c ON c.id = bi.camera_id
JOIN zones z ON z.id = bi.zone_id
ORDER BY bi.entered_at DESC
LIMIT 10;
```

### Check Purchase Count:

```sql
-- Total purchases today
SELECT COUNT(*) as purchase_count
FROM billing_interactions
WHERE DATE(entered_at) = CURRENT_DATE;

-- Purchases by hour today
SELECT 
    DATE_TRUNC('hour', entered_at) as hour,
    COUNT(*) as purchases
FROM billing_interactions
WHERE DATE(entered_at) = CURRENT_DATE
GROUP BY hour
ORDER BY hour;
```

---

## Quick All-in-One Setup Script

```sql
DO $ 
DECLARE
    v_camera_id uuid;
    v_zone_id uuid;
    v_rule_id uuid;
BEGIN
    -- Get first active camera
    SELECT id INTO v_camera_id 
    FROM cameras 
    WHERE is_active = true 
    ORDER BY created_at 
    LIMIT 1;
    
    IF v_camera_id IS NULL THEN
        RAISE EXCEPTION 'No active camera found!';
    END IF;
    
    RAISE NOTICE 'Using camera ID: %', v_camera_id;
    
    -- Check if billing zone exists
    SELECT id INTO v_zone_id
    FROM zones
    WHERE camera_id = v_camera_id 
    AND zone_type = 'billing_zone'
    LIMIT 1;
    
    IF v_zone_id IS NOT NULL THEN
        RAISE NOTICE 'Billing zone already exists: %', v_zone_id;
    ELSE
        -- Create billing zone
        INSERT INTO zones (
            id, camera_id, name, zone_type, shape, polygon, is_active, created_at, updated_at
        ) VALUES (
            gen_random_uuid(), v_camera_id, 'Billing Counter', 'billing_zone', 'polygon',
            '{"points": [[30, 30], [70, 30], [70, 70], [30, 70]]}'::jsonb, true, NOW(), NOW()
        )
        RETURNING id INTO v_zone_id;
        
        RAISE NOTICE 'Created billing zone: %', v_zone_id;
    END IF;
    
    -- Check if rule exists
    SELECT id INTO v_rule_id
    FROM rules
    WHERE zone_id = v_zone_id 
    AND rule_type = 'billing_interaction'
    LIMIT 1;
    
    IF v_rule_id IS NOT NULL THEN
        RAISE NOTICE 'Billing rule already exists: %', v_rule_id;
    ELSE
        -- Create billing interaction rule
        INSERT INTO rules (
            id, camera_id, zone_id, name, rule_type, config, 
            cooldown_seconds, severity, dwell_threshold_seconds, is_enabled, created_at, updated_at
        ) VALUES (
            gen_random_uuid(), v_camera_id, v_zone_id, 'Billing Counter Interaction',
            'billing_interaction', '{"threshold_seconds": 5}'::jsonb,
            30, 'info', 5, true, NOW(), NOW()
        )
        RETURNING id INTO v_rule_id;
        
        RAISE NOTICE 'Created billing rule: %', v_rule_id;
    END IF;
    
    RAISE NOTICE '✅ Setup complete!';
    RAISE NOTICE 'Camera ID: %', v_camera_id;
    RAISE NOTICE 'Zone ID: %', v_zone_id;
    RAISE NOTICE 'Rule ID: %', v_rule_id;
    
END $;
```

---

## Summary

✅ **Step 1:** Create `billing_zone` polygon around checkout counter  
✅ **Step 2:** Create `billing_interaction` rule with 5-second threshold  
✅ **Step 3:** Verify setup with verification query  
✅ **Step 4:** Monitor billing_interactions table  

**Purchase tracking is now active!** 🎉
