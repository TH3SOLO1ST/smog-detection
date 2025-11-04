# Islamabad Smog Detection System - Simple User Guide

> **Made for everyone!** You don't need to be technical to use this system.

## 🌫️ Welcome to Your Air Quality Monitor

This system helps you understand the air quality in Islamabad, Pakistan. It shows you if the air is clean or polluted, and tells you how to stay healthy.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Your First Day](#your-first-day)
3. [Understanding the Dashboard](#understanding-the-dashboard)
4. [Daily Operations](#daily-operations)
5. [What the Colors Mean](#what-the-colors-mean)
6. [Getting Help](#getting-help)

---

## 🚀 Getting Started

### Step 1: Install the System

**Windows Users:**
1. Double-click the `install.py` file
2. Follow the instructions on screen
3. Wait for installation to finish (may take 10-15 minutes)
4. Look for "Islamabad Smog Monitor" on your desktop

**Mac/Linux Users:**
1. Open Terminal
2. Go to the smog-detection folder
3. Type: `python install.py`
4. Follow the instructions

### Step 2: Start the System

**Easy way:** Double-click the desktop icon "Islamabad Smog Monitor"

**Manual way:**
1. Open the smog-detection folder
2. Double-click `start_system.bat` (Windows) or run `python src/web_interface/app.py`

### Step 3: Open Your Dashboard

1. Open your web browser (Chrome, Firefox, Safari, etc.)
2. Go to: `http://localhost:5000`
3. Bookmark this page for easy access

---

## 🌟 Your First Day

### What You'll See

When you first open the dashboard, you'll see:

![Dashboard Overview](images/dashboard-overview.png)

1. **Big Color Status** - Shows if air is good or bad
2. **Numbers** - Pollution levels for today
3. **Map** - Shows air quality in different areas
4. **Chart** - Shows last 30 days of air quality

### How to Read It

**Top Section - Current Status:**
- 😊 **Good** - Air is clean, safe for everyone
- 😐 **Moderate** - Okay for most people
- 😷 **Unhealthy for Sensitive Groups** - Be careful if you have asthma or are elderly
- 🤢 **Unhealthy** - Everyone should limit outdoor activities

**Map Section:**
- Green dots = Clean air
- Yellow dots = Moderate pollution
- Red dots = High pollution
- Click any dot to see details

---

## 📊 Understanding the Dashboard

### Main Parts of Your Dashboard

#### 1. Current Air Quality Box
```
😊 GOOD
Air Quality Index: 45
Updated: 2:30 PM
```

**What this means:**
- The emoji shows how healthy the air is
- The number (0-500) shows exact air quality
- Lower numbers = Better air

#### 2. Pollutant Cards
```
PM2.5: 25.4 µg/m³    GOOD
NO₂: 18.2 µg/m³      GOOD
SO₂: 8.1 µg/m³       GOOD
```

**What these are:**
- **PM2.5** - Tiny particles from smoke and dust
- **NO₂** - Gas from cars and factories
- **SO₂** - Gas from burning fuel
- **O₃** - Ground-level ozone
- **CO** - Carbon monoxide

**Safe levels:**
- Green = Safe for everyone
- Yellow = Mostly safe
- Orange = Be careful
- Red = Stay indoors if possible

#### 3. Interactive Map
- Zoom in/out with +/- buttons
- Click colored circles for details
- Different colors = Different pollution levels

#### 4. Health Recommendations
```
🏥 Health Advice:
✅ Enjoy outdoor activities
✅ Safe for children and elderly
✅ Great day for exercise
```

This section tells you:
- Is it safe to go outside?
- Should you avoid exercise?
- Do you need to wear a mask?

---

## 🔄 Daily Operations

### Checking Today's Air Quality

1. **Open the dashboard** (bookmark it for easy access)
2. **Look at the big status** at the top
3. **Check the health advice** section
4. **Plan your day** based on recommendations

### What to Do Each Day

**If air is GOOD (😊):**
- ✅ Walk, jog, or exercise outside
- ✅ Open windows for fresh air
- ✅ Children can play outside

**If air is MODERATE (😐):**
- ✅ Normal activities are fine
- ⚠️ Sensitive people should watch for symptoms
- ✅ Windows can stay open

**If air is UNHEALTHY (😷 or 🤢):**
- ❌ Avoid outdoor exercise
- ❌ Keep windows closed
- ✅ Use air purifier if you have one
- ✅ Wear mask if you must go outside

### Setting Up Alerts

1. Click **Settings** (gear icon ⚙️)
2. Go to **Alerts & Reports** tab
3. Enter your email address
4. Choose when to get alerts
5. Click **Save Settings**

### Getting Daily Reports

**Automatic Reports:**
1. Go to Settings → Alerts & Reports
2. Turn ON "Daily Report"
3. Choose time (8:00 AM is good)
4. Add your email
5. Save settings

**Manual Reports:**
1. On main dashboard, click **"Daily Report"** button
2. Wait 30 seconds
3. PDF will download automatically

---

## 🎨 What the Colors Mean

### Air Quality Colors

| Color | Range | What It Means | What To Do |
|-------|-------|---------------|------------|
| 🟢 **Green** | 0-50 | Air is clean | Enjoy outdoor activities |
| 🟡 **Yellow** | 51-100 | Moderate air quality | Normal activities okay |
| 🟠 **Orange** | 101-150 | Unhealthy for sensitive groups | Limit outdoor exercise |
| 🔴 **Red** | 151-200 | Unhealthy | Avoid outdoor activities |
| 🟣 **Purple** | 201-300 | Very unhealthy | Stay indoors |
| 🔴 **Maroon** | 301+ | Hazardous | Emergency - stay indoors |

### Map Colors

- **Green circles** = Clean air areas
- **Yellow circles** = Moderate pollution
- **Orange circles** = High pollution
- **Red circles** = Very high pollution

### Status Emojis

- 😊 **Good** - Perfect day to be outside
- 😐 **Moderate** - Generally okay
- 😷 **Sensitive** - Be careful if you have health issues
- 🤢 **Unhealthy** - Avoid outside activities
- 😵 **Very Unhealthy** - Stay inside
- ☠️ **Hazardous** - Emergency level

---

## 📱 Using on Your Phone

### Mobile View

The dashboard works great on phones!

1. Open browser on your phone
2. Go to `http://localhost:5000`
3. Or `http://[your-computer-ip]:5000`

**Note:** Your computer must be on for this to work.

### What's Different on Mobile

- **Bigger buttons** for easy tapping
- **Simplified charts** for small screens
- **Swipeable map** for easy navigation
- **Quick status** at the top

---

## ⚙️ Basic Settings

### Changing What You See

1. Click **Settings** (gear icon ⚙️)
2. Choose what you want to change:
   - **Data Sources** - Turn on/off different data
   - **Alerts** - Set up email notifications
   - **Region** - Change the area you're monitoring
   - **Processing** - How data is processed

### Important Settings for Non-Technical Users

**Email Alerts:**
- Turn on if you want daily emails
- Good for office managers or schools
- Set quiet hours so you don't get alerts at night

**Data Sources:**
- Keep all ON for best results
- Each source provides different information

**Region Settings:**
- Default is perfect for Islamabad
- Only change if you need different area

---

## 🆘 Getting Help

### Common Problems

**"Dashboard won't load"**
1. Make sure the system is running (look for the system icon)
2. Try refreshing the page (F5 or Ctrl+R)
3. Check you're using `http://localhost:5000`

**"Map shows blank"**
1. Check your internet connection
2. Try refreshing the page
3. Wait 1-2 minutes for data to load

**"Numbers seem wrong"**
1. Data updates every few hours
2. Check the "Last Update" time
3. Different stations might show different numbers

**"Email alerts not working"**
1. Check your email address in Settings
2. Click "Test Email" button
3. Check spam/junk folder

### When to Get Technical Help

Contact support if:
- System won't install
- Dashboard never loads
- No data shows for more than 24 hours
- You see error messages

**What to tell support:**
1. What you were trying to do
2. What you saw on screen
3. Any error messages
4. Your computer type (Windows/Mac)

### Quick Fixes That Solve Most Problems

1. **Refresh the page** (F5 or Ctrl+R)
2. **Restart the system** (close and reopen)
3. **Check internet connection**
4. **Wait 5 minutes and try again**
5. **Clear browser cache** (Ctrl+Shift+Delete)

---

## 📚 Learning More

### Understanding the Numbers

**PM2.5 (Most Important)**
- Tiny particles that go deep into lungs
- Comes from smoke, dust, traffic
- WHO says: Keep below 15 µg/m³

**NO₂ (Nitrogen Dioxide)**
- From cars, trucks, factories
- Can make asthma worse
- WHO says: Keep below 40 µg/m³

**SO₂ (Sulfur Dioxide)**
- From burning coal and oil
- Can irritate breathing
- WHO says: Keep below 50 µg/m³

### Why Air Quality Changes

**Better Air Quality When:**
- After rain (washes pollution away)
- Windy days (blows pollution away)
- Less traffic (holidays, weekends)

**Worse Air Quality When:**
- No wind for several days
- Cold weather (traps pollution near ground)
- Heavy traffic
- Industrial activity
- Farm fires in region

### Health Tips

**For Everyone:**
- Check air quality before exercise
- Keep windows closed on bad days
- Use air purifiers if possible
- Wear masks on very polluted days

**For Sensitive People:**
- Asthma, heart disease, elderly, children
- Check air quality daily
- Avoid outdoor exercise on bad days
- Keep rescue medication handy

---

## 🎯 Quick Reference

### Daily Checklist (2 Minutes)

1. **Open dashboard** - 30 seconds
2. **Check status color** - 15 seconds
3. **Read health advice** - 30 seconds
4. **Plan your day** - 45 seconds

### When to Take Action

| AQI Range | What to Do |
|-----------|------------|
| 0-50 (Green) | Normal activities |
| 51-100 (Yellow) | Normal activities |
| 101-150 (Orange) | Limit outdoor exercise |
| 151-200 (Red) | Stay indoors, keep windows closed |
| 201+ (Purple/Maroon) | Emergency - stay indoors |

### Emergency Contacts

**For Health Issues:**
- Call your doctor for breathing problems
- Hospital emergency: 1122 (Pakistan)

**For Technical Issues:**
- Email: support@islamabad-smog.gov.pk
- Phone: [Support Number]

**For Air Quality Complaints:**
- Environmental Agency: [Local Number]
- Pakistan EPA: [EPA Number]

---

## 📝 Frequently Asked Questions

**Q: Is the system always accurate?**
A: We use multiple data sources for accuracy. Data is checked and validated automatically.

**Q: How often does data update?**
A: Most data updates every hour. Some satellite data updates once per day.

**Q: Can I check air quality for other cities?**
A: This system is designed specifically for Islamabad and surrounding areas.

**Q: Do I need to pay for this service?**
A: Basic air quality monitoring is free. Advanced features may have costs.

**Q: Can I share this with others?**
A: Yes! Use the Share button to send the dashboard link to family and friends.

**Q: What if I don't understand something?**
A: Look at the color codes - green is good, red is bad. When in doubt, treat yellow/red as "be careful."

---

## 🎉 Congratulations!

You now know how to:
✅ Install and start the system
✅ Read air quality information
✅ Understand what the colors mean
✅ Take action based on air quality
✅ Get help when you need it

**Remember:**
- Green = Good to go outside
- Yellow = Be careful
- Orange/Red = Stay indoors if possible

Check the air quality every day, especially before planning outdoor activities!

---

*This guide is written for non-technical users. For advanced features and technical details, see the full documentation.*