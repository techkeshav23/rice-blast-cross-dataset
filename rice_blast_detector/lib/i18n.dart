import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Lightweight app localization (English + Hindi) with a persisted choice.
/// Wrap MaterialApp in a ValueListenableBuilder(valueListenable: I18n.lang)
/// so the whole tree rebuilds instantly on language change.
class I18n {
  static const _key = 'app_lang_v1';
  static final ValueNotifier<String> lang = ValueNotifier<String>('en');

  static Future<void> load() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      lang.value = prefs.getString(_key) ?? 'en';
    } catch (_) {
      lang.value = 'en';
    }
  }

  static Future<void> set(String code) async {
    lang.value = code;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_key, code);
    } catch (_) {}
  }

  static String t(String key) {
    final entry = _s[key];
    if (entry == null) return key;
    return entry[lang.value] ?? entry['en'] ?? key;
  }

  static const Map<String, Map<String, String>> _s = {
    // ---- nav / shell ----
    'nav_scan': {'en': 'Scan', 'hi': 'स्कैन'},
    'nav_guide': {'en': 'Guide', 'hi': 'गाइड'},
    'nav_guide_full': {'en': 'Field Guide', 'hi': 'फ़ील्ड गाइड'},
    'nav_history': {'en': 'History', 'hi': 'इतिहास'},
    'nav_how': {'en': 'How it works', 'hi': 'यह कैसे काम करता है'},
    'nav_results': {'en': 'Results', 'hi': 'नतीजे'},
    'nav_about': {'en': 'About', 'hi': 'परिचय'},
    'nav_settings': {'en': 'Settings', 'hi': 'सेटिंग'},
    'grp_scan': {'en': 'SCAN', 'hi': 'स्कैन'},
    'grp_project': {'en': 'PROJECT', 'hi': 'प्रोजेक्ट'},
    'brand_tagline': {'en': 'Rice disease scanner', 'hi': 'धान रोग स्कैनर'},
    'offline_footer': {
      'en': 'Works offline · v1.0 demo',
      'hi': 'ऑफ़लाइन चलता है · v1.0 डेमो'
    },

    // ---- severity ----
    'sev_high': {'en': 'High severity', 'hi': 'गंभीर'},
    'sev_medium': {'en': 'Moderate', 'hi': 'मध्यम'},
    'sev_none': {'en': 'Healthy', 'hi': 'स्वस्थ'},
    'sev_unknown': {'en': 'Unknown', 'hi': 'अज्ञात'},

    // ---- home ----
    'home_eyebrow': {'en': 'RICE LEAF DIAGNOSIS', 'hi': 'धान की पत्ती की जाँच'},
    'home_title': {
      'en': 'Detect rice\nblast early.',
      'hi': 'राइस ब्लास्ट को\nसमय रहते पहचानें।'
    },
    'home_sub': {
      'en':
          'Capture a paddy leaf and get an instant read — blast, brown spot, or healthy. Built to stay accurate across fields and cameras.',
      'hi':
          'धान की पत्ती की फोटो लें और तुरंत नतीजा पाएँ — ब्लास्ट, ब्राउन स्पॉट या स्वस्थ। अलग-अलग खेत और कैमरों पर सटीक रहने के लिए बनाया गया।'
    },
    'offline': {'en': 'Works offline', 'hi': 'बिना इंटरनेट चलता है'},
    'btn_scan': {'en': 'Scan a leaf', 'hi': 'पत्ती स्कैन करें'},
    'btn_gallery': {'en': 'Upload from gallery', 'hi': 'गैलरी से चुनें'},
    'hero_frame': {'en': 'Point at a rice leaf', 'hi': 'धान की पत्ती पर लगाएँ'},
    'stat_diseases': {'en': 'Diseases', 'hi': 'रोग'},
    'stat_offline': {'en': 'Offline', 'hi': 'ऑफ़लाइन'},
    'stat_fast': {'en': 'Per scan', 'hi': 'प्रति स्कैन'},
    'detects': {'en': 'WHAT IT DETECTS', 'hi': 'क्या-क्या पहचानता है'},
    'tip': {
      'en':
          'Tip: fill the frame with one leaf in good daylight. Sharp, well-lit photos give the most reliable reading.',
      'hi':
          'सुझाव: दिन के उजाले में एक ही पत्ती को पूरे फ्रेम में लें। साफ़ और अच्छी रोशनी वाली फोटो सबसे भरोसेमंद नतीजा देती है।'
    },
    'analysing_title': {'en': 'Reading the leaf', 'hi': 'पत्ती पढ़ी जा रही है'},
    'analysing_sub': {
      'en': 'Analysing texture and lesions…',
      'hi': 'बनावट और धब्बों का विश्लेषण…'
    },

    // ---- result ----
    'res_title': {'en': 'Diagnosis', 'hi': 'निदान'},
    'res_mock_banner': {
      'en':
          'Simulated result — mock model (demo). Real predictions arrive once the trained model is connected.',
      'hi':
          'यह नमूना नतीजा है — मॉक मॉडल (डेमो)। असली भविष्यवाणी तब आएगी जब प्रशिक्षित मॉडल जुड़ेगा।'
    },
    'res_mock': {'en': 'MOCK', 'hi': 'मॉक'},
    'res_gradcam_eyebrow': {
      'en': 'MODEL ATTENTION · GRAD-CAM',
      'hi': 'मॉडल का ध्यान · ग्रेड-कैम'
    },
    'res_gradcam_caption': {
      'en':
          'Warm areas = where a model would focus. Illustrative until the trained model is connected.',
      'hi':
          'गर्म रंग = जहाँ मॉडल ध्यान देगा। असली मॉडल जुड़ने तक यह केवल उदाहरण है।'
    },
    'gradcam_chip': {'en': 'Illustrative heatmap · demo', 'hi': 'उदाहरण हीटमैप · डेमो'},
    'confidence': {'en': 'CONFIDENCE', 'hi': 'विश्वास'},
    'symptoms': {'en': 'SYMPTOMS', 'hi': 'लक्षण'},
    'management': {'en': 'MANAGEMENT', 'hi': 'प्रबंधन'},
    'res_lowconf': {
      'en':
          'Low confidence — this leaf may fall outside the conditions the model knows. Retake in daylight, fill the frame, or consult an expert.',
      'hi':
          'कम विश्वास — यह पत्ती मॉडल की जानी-पहचानी स्थितियों से बाहर हो सकती है। दिन के उजाले में दोबारा फोटो लें, फ्रेम भरें, या विशेषज्ञ से सलाह लें।'
    },
    'sources': {
      'en': 'Sources: IRRI Rice Knowledge Bank · ICAR · TNAU Agritech',
      'hi': 'स्रोत: IRRI राइस नॉलेज बैंक · ICAR · TNAU एग्रीटेक'
    },

    // ---- library ----
    'lib_eyebrow': {'en': 'KNOW THE CONDITIONS', 'hi': 'स्थितियाँ जानें'},
    'lib_title': {
      'en': 'Three things a\npaddy leaf can be.',
      'hi': 'धान की पत्ती\nतीन में से एक हो सकती है।'
    },

    // ---- history ----
    'hist_eyebrow': {'en': 'YOUR SCANS', 'hi': 'आपके स्कैन'},
    'hist_title': {'en': 'Recent readings', 'hi': 'हाल के नतीजे'},
    'hist_clear': {'en': 'Clear', 'hi': 'साफ़ करें'},
    'hist_empty_title': {'en': 'No scans yet', 'hi': 'अभी कोई स्कैन नहीं'},
    'hist_empty_sub': {
      'en': 'Analysed leaves will show up here.',
      'hi': 'जाँची गई पत्तियाँ यहाँ दिखेंगी।'
    },

    // ---- how it works ----
    'how_eyebrow': {'en': 'THE RESEARCH', 'hi': 'शोध'},
    'how_title': {
      'en': 'Staying accurate on\nunseen fields.',
      'hi': 'अनदेखे खेतों पर भी\nसटीक बने रहना।'
    },
    'how_s1_t': {'en': 'The problem', 'hi': 'समस्या'},
    'how_s1_b': {
      'en':
          'Rice-disease models often score ~99% on their own training dataset, but accuracy can collapse (sometimes to ~32%) on leaves from a different field, camera or lighting. They learn the dataset, not the disease.',
      'hi':
          'धान-रोग मॉडल अक्सर अपने प्रशिक्षण डेटासेट पर ~99% सटीक होते हैं, पर किसी अलग खेत, कैमरे या रोशनी की पत्ती पर सटीकता गिर सकती है (कभी-कभी ~32% तक)। वे रोग नहीं, डेटासेट को सीख लेते हैं।'
    },
    'how_s2_t': {'en': 'Our approach', 'hi': 'हमारा तरीका'},
    'how_s2_b': {
      'en':
          'Train on one rice dataset (Paddy Doctor) and test on a separate, never-seen external dataset. A lightweight domain-generalization method (MixStyle) is added to reduce the drop — no access to the test domain during training.',
      'hi':
          'एक डेटासेट (Paddy Doctor) पर प्रशिक्षण दें और एक अलग, कभी न देखे गए डेटासेट पर परीक्षण करें। गिरावट कम करने के लिए एक हल्की डोमेन-जनरलाइज़ेशन विधि (MixStyle) जोड़ी जाती है — प्रशिक्षण के दौरान परीक्षण डेटा का उपयोग नहीं।'
    },
    'how_s3_t': {'en': 'How we measure it', 'hi': 'हम इसे कैसे मापते हैं'},
    'how_s3_b': {
      'en':
          'We report Macro-F1 (not just accuracy) and the internal-to-external gap. A smaller gap means the model generalizes better to real fields.',
      'hi':
          'हम केवल सटीकता नहीं, Macro-F1 और अपने-से-बाहरी अंतर बताते हैं। जितना कम अंतर, मॉडल असली खेतों पर उतना बेहतर।'
    },
    'how_s4_t': {'en': 'Why it matters', 'hi': 'यह क्यों ज़रूरी है'},
    'how_s4_b': {
      'en':
          'Real paddy fields never look like clean lab photos. A model that survives the domain shift is one a farmer can actually trust.',
      'hi':
          'असली धान के खेत साफ़ लैब फोटो जैसे नहीं दिखते। जो मॉडल इस बदलाव में टिकता है, किसान उसी पर भरोसा कर सकता है।'
    },
    'how_note': {
      'en':
          'Focus: Rice Blast. Brown Spot and Healthy are included as context classes for the prototype.',
      'hi':
          'मुख्य लक्ष्य: राइस ब्लास्ट। ब्राउन स्पॉट और स्वस्थ को प्रोटोटाइप के लिए संदर्भ वर्ग के रूप में शामिल किया गया है।'
    },

    // ---- metrics ----
    'met_eyebrow': {'en': 'GENERALIZATION GAP', 'hi': 'जनरलाइज़ेशन अंतर'},
    'met_title': {
      'en': 'Does it hold up on\nan unseen dataset?',
      'hi': 'क्या यह अनदेखे\nडेटासेट पर टिकता है?'
    },
    'met_sub': {
      'en':
          'Macro-F1 on the training dataset vs an unseen external dataset. The smaller the drop, the better the model generalizes.',
      'hi':
          'प्रशिक्षण डेटासेट बनाम अनदेखे बाहरी डेटासेट पर Macro-F1। जितनी कम गिरावट, मॉडल उतना बेहतर जनरलाइज़ करता है।'
    },
    'met_inset': {'en': 'IN-SET', 'hi': 'अपना सेट'},
    'met_unseen': {'en': 'UNSEEN', 'hi': 'अनदेखा'},
    'met_gap': {'en': 'GAP', 'hi': 'अंतर'},
    'met_baseline': {'en': 'Baseline', 'hi': 'बेसलाइन'},
    'met_ours': {'en': '+ MixStyle (ours)', 'hi': '+ MixStyle (हमारा)'},
    'met_takeaway': {
      'en':
          'Real test (Paddy Doctor → unseen Philippines): MixStyle shrinks the gap from 22.9 to 19.6 Macro-F1 points and lifts external Macro-F1 by +2.3 — the model travels better.',
      'hi':
          'असली परीक्षण (Paddy Doctor → अनदेखा Philippines): MixStyle अंतर को 22.9 से घटाकर 19.6 Macro-F1 अंक करता है और बाहरी Macro-F1 +2.3 बढ़ाता है — मॉडल बेहतर सफ़र करता है।'
    },
    'met_illustrative': {
      'en': 'Real result: Paddy Doctor → unseen Philippines dataset (single-seed).',
      'hi': 'असली नतीजा: Paddy Doctor → अनदेखा Philippines डेटासेट (सिंगल-सीड)।'
    },

    // ---- about ----
    'about_eyebrow': {'en': 'THE PROJECT', 'hi': 'प्रोजेक्ट'},
    'about_body': {
      'en':
          'A prototype for Rice Blast detection with a research focus on cross-dataset generalization — staying accurate on leaves from fields, cameras and conditions the model has never seen.',
      'hi':
          'राइस ब्लास्ट पहचान का एक प्रोटोटाइप, जिसका शोध-केंद्र क्रॉस-डेटासेट जनरलाइज़ेशन है — उन खेतों, कैमरों और स्थितियों की पत्तियों पर भी सटीक रहना जो मॉडल ने कभी नहीं देखीं।'
    },
    'about_team': {'en': 'TEAM', 'hi': 'टीम'},
    'about_guide': {'en': 'GUIDE', 'hi': 'मार्गदर्शक'},
    'about_institute': {'en': 'INSTITUTE', 'hi': 'संस्थान'},
    'about_institute2': {
      'en': 'Dept. of CSE · Session 2026-27',
      'hi': 'CSE विभाग · सत्र 2026-27'
    },
    'about_disclaimer': {
      'en':
          'Decision support only. This app does not replace advice from agricultural experts.',
      'hi':
          'केवल निर्णय-सहायता। यह ऐप कृषि विशेषज्ञों की सलाह की जगह नहीं लेता।'
    },

    // ---- settings ----
    'settings_eyebrow': {'en': 'LANGUAGE', 'hi': 'भाषा'},
    'settings_title': {'en': 'Choose your language', 'hi': 'अपनी भाषा चुनें'},
    'settings_hint': {
      'en': 'The whole app, including disease guidance, switches instantly.',
      'hi': 'पूरा ऐप, रोग सलाह समेत, तुरंत बदल जाता है।'
    },
  };
}

/// Shorthand.
String tr(String key) => I18n.t(key);
