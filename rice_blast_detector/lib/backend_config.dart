/// Configuration for the optional FastAPI backend.
///
/// This app is offline-first. The backend is a best-effort enhancement that is
/// only used when the device is online and [kBackendEnabled] is `true`. Every
/// network call is wrapped so it fails gracefully — the app must remain fully
/// functional even if the backend is unreachable.
library;

/// Base URL of the deployed backend.
///
/// After deploying to Hugging Face Spaces, replace the placeholder below with
/// your Space's public URL, e.g. `https://your-name-rice-blast.hf.space`.
const String kBackendBaseUrl = 'https://REPLACE-WITH-YOUR-SPACE.hf.space';

/// Master switch for backend usage. Set to `false` to disable all remote calls.
const bool kBackendEnabled = true;

/// Timeout applied to every backend request. Kept short so the offline-first
/// UI never blocks for long on a slow or unreachable server.
const Duration kBackendTimeout = Duration(seconds: 20);
