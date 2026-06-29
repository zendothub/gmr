# Purchase Tracking Setup Guide

## How to Enable Purchase Count in Analytics Dashboard

Currently, your purchase count is **0** because no billing interactions are being tracked. Here's how to set it up:

---

## Step 1: Create a Billing Zone

### Via Frontend (Recommended):

1. **Go to Cameras page** → Select your camera
2. **Click "Add Zone"** or "Edit Zones"
3. **Draw a polygon** around your **billing/checkout counter area**
4. **Set the zone details:**
   - **Name**: `Billing Counter` or `Checkout Area`
   - **Zone Type**: `billing_zone` (important!)
   - **Shape**: `polygon`
5. **Save the zone**

### Via API (if needed):

```bash
POST /api/cameras/{camera_id}/zones
{
  "name": "Billing Counter",
  "zone_type": "billing_zone",
  "shape": "polygon",
  "polygon": {
    "points": [
      [100, 200],  # top-left corner (x, y in pixels)
      [500, 200],  # top-right
      [500, 600],  # bottom-right
      [100, 600]   # bottom-left
    ]
  },
  "is_active": true
}
```

---

## Step 2: Create a Billing Interaction Rule

### Via Frontend:

1. **Go to Rules page** (or create via API)
2. **Click "Add Rule"**
3. **Fill in:**
   - **Name**: `Billing Counter Interaction`
   - **Rule Type**: `billing_interaction` (important!)
   - **Zone**: Select the billing zone you just created
   - **Camera**: Select your camera
   - **Parameters**: 
     ```json
     {
       "threshold_seconds": 5
     }
     ```
     (Person must stay in zone for 5 seconds to count as purchase)
4. **Save the rule**

### Via API:

```bash
POST /api/rules
{
  "name": "Billing Counter Interaction",
  "rule_type": "billing_interaction",
  "zone_id": "<your-billing-zone-id>",
  "camera_id": "<your-camera-id>",
  "parameters": {
    "threshold_seconds": 5
  },
  "is_active": true
}
```

---

## Step 3: Test the Setup

1. **Have someone stand in the billing zone** for 5+ seconds
2. **Check if billing interaction was created:**

```sql
SELECT * FROM billing_interactions 
ORDER BY entered_at DESC 
LIMIT 10;
```

3. **Check the analytics dashboard**
   - Refresh the page
   - Purchase count should now be > 0
   - Conversion rate should calculate automatically

---

## How It Works

### When a Person Enters the Billing Zone:

1. **Camera detects person** → Creates TrackSession
2. **Person enters billing zone** → Zone membership tracked
3. **Person dwells for 5+ seconds** → Rule fires
4. **BillingInteraction created**:
   ```python
   BillingInteraction(
       camera_id=...,
       zone_id=...,
       person_identity_id=...,
       track_session_id=...,
       entered_at=<when they entered>,
       exited_at=<when they left>,
       dwell_seconds=<how long they stayed>
   )
   ```
5. **Analytics counts this as a purchase** ✅

---

## Available Rule Types

Your system supports 6 rule types:

| Rule Type | Description | Creates BillingInteraction? |
|-----------|-------------|----------------------------|
| `line_crossing` | Person crosses a line | ❌ No |
| `zone_dwell` | Person stays in zone > threshold | ❌ No |
| `billing_interaction` | Person in billing zone > threshold | ✅ **YES** |
| `queue_count` | Count people in queue zone | ❌ No |
| `possible_purchase` | Dwell in purchase intent area | ❌ No |
| `restricted_zone` | Person enters restricted area | ❌ No |

**Only `billing_interaction` creates purchase records!**

---

## Troubleshooting

### Purchase count still 0 after setup?

1. **Check zone is active:**
   ```sql
   SELECT * FROM zones WHERE zone_type = 'billing_zone';
   ```

2. **Check rule is active:**
   ```sql
   SELECT * FROM rules WHERE rule_type = 'billing_interaction';
   ```

3. **Check camera worker is running:**
   - Look at server logs
   - Should see: `RuleEvaluator` processing frames

4. **Check billing interactions table:**
   ```sql
   SELECT COUNT(*) FROM billing_interactions;
   ```

5. **Test with a real person:**
   - Stand in billing zone for 5+ seconds
   - Check logs for "billing_interaction" event
   - Refresh analytics dashboard

---

## Quick Test Command

```bash
# Check if everything is set up
SELECT 
    c.name as camera_name,
    z.name as zone_name,
    z.zone_type,
    r.name as rule_name,
    r.rule_type,
    r.is_active as rule_active
FROM cameras c
LEFT JOIN zones z ON z.camera_id = c.id AND z.zone_type = 'billing_zone'
LEFT JOIN rules r ON r.zone_id = z.id AND r.rule_type = 'billing_interaction'
WHERE c.is_active = true;
```

Expected output:
- Should show your camera
- Should show billing zone
- Should show billing_interaction rule
- All should be active

---

## Summary

✅ **Create** a `billing_zone` type zone around checkout area  
✅ **Create** a `billing_interaction` type rule for that zone  
✅ **Test** by having someone stand in the zone  
✅ **Verify** BillingInteraction records are created  
✅ **Check** analytics dashboard shows purchase count  

That's it! Your purchase tracking will now work automatically.
