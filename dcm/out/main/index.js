"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const electron = require("electron");
const path = require("path");
const utils = require("@electron-toolkit/utils");
const fs = require("fs");
const argon2 = require("argon2");
const child_process = require("child_process");
const json2csv = require("json2csv");
function _interopNamespaceDefault(e) {
  const n = Object.create(null, { [Symbol.toStringTag]: { value: "Module" } });
  if (e) {
    for (const k in e) {
      if (k !== "default") {
        const d = Object.getOwnPropertyDescriptor(e, k);
        Object.defineProperty(n, k, d.get ? d : {
          enumerable: true,
          get: () => e[k]
        });
      }
    }
  }
  n.default = e;
  return Object.freeze(n);
}
const path__namespace = /* @__PURE__ */ _interopNamespaceDefault(path);
const argon2__namespace = /* @__PURE__ */ _interopNamespaceDefault(argon2);
const iconIco = path.join(__dirname, "../../resources/icon.ico");
const iconPng = path.join(__dirname, "../../resources/icon.png");
const usersFilePath = path.join(electron.app.getPath("userData"), "users.json");
const createUser = (overrides = {}) => ({
  username: "",
  passwordHash: "",
  serialNumber: "",
  modes: {
    VOO: {
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0
    },
    AOO: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0
    },
    VVI: {
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0
    },
    AAI: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0
    },
    DDDR: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      rateFactor: 0,
      avDelay: 0,
      reactionTime: 0,
      recoveryTime: 0,
      activityThreshold: 4
    },
    DDD: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      avDelay: 0
    },
    AOOR: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      rateFactor: 0,
      reactionTime: 0,
      recoveryTime: 0,
      activityThreshold: 4
    },
    AAIR: {
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      rateFactor: 0,
      reactionTime: 0,
      recoveryTime: 0,
      activityThreshold: 4
    },
    VOOR: {
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      rateFactor: 0,
      reactionTime: 0,
      recoveryTime: 0,
      activityThreshold: 4
    },
    VVIR: {
      ventricularAmplitude: 0,
      ventricularPulseWidth: 0,
      ventricularRefractoryPeriod: 0,
      atrialAmplitude: 0,
      atrialPulseWidth: 0,
      atrialRefractoryPeriod: 0,
      lowerRateLimit: 0,
      upperRateLimit: 0,
      rateFactor: 0,
      reactionTime: 0,
      recoveryTime: 0,
      activityThreshold: 4
    }
  },
  lastUsedMode: "OFF",
  ...overrides
});
const parameterHistoryPath$1 = path__namespace.join(__dirname, "../../parameterHistory.json");
async function ensureUsersFile(filePath) {
  try {
    await fs.promises.access(filePath);
  } catch (error) {
    if (error.code === "ENOENT") {
      await fs.promises.writeFile(filePath, JSON.stringify([]));
    } else {
      throw error;
    }
  }
}
async function ensureDirectoryExists(filePath) {
  const dir = path__namespace.dirname(filePath);
  try {
    await fs.promises.access(dir);
  } catch (error) {
    if (error.code === "ENOENT") {
      await fs.promises.mkdir(dir, { recursive: true });
    } else {
      throw error;
    }
  }
}
async function ensureParameterHistoryFile(filePath) {
  await ensureDirectoryExists(filePath);
  try {
    await fs.promises.access(filePath);
  } catch (error) {
    if (error.code === "ENOENT") {
      await fs.promises.writeFile(filePath, JSON.stringify([]));
    } else {
      throw error;
    }
  }
}
async function getUsers(filePath) {
  const data = await fs.promises.readFile(filePath, "utf-8");
  return JSON.parse(data);
}
async function saveUser(users) {
  await fs.promises.writeFile(usersFilePath, JSON.stringify(users, null, 2));
}
async function addUserToHistory(username, serialNumber) {
  await ensureParameterHistoryFile(parameterHistoryPath$1);
  const data = await fs.promises.readFile(parameterHistoryPath$1, "utf-8");
  const history = JSON.parse(data);
  const registrationDate = (/* @__PURE__ */ new Date()).toISOString();
  history.push({ username, serialNumber, registrationDate, loginHistory: [], parameterChanges: [] });
  await fs.promises.writeFile(parameterHistoryPath$1, JSON.stringify(history, null, 2));
}
async function logUserLogin(username) {
  await ensureParameterHistoryFile(parameterHistoryPath$1);
  const data = await fs.promises.readFile(parameterHistoryPath$1, "utf-8");
  const history = JSON.parse(data);
  const userHistory = history.find((entry) => entry.username === username);
  if (userHistory) {
    const loginDate = (/* @__PURE__ */ new Date()).toISOString();
    userHistory.loginHistory.push({ loginDate });
    await fs.promises.writeFile(parameterHistoryPath$1, JSON.stringify(history, null, 2));
  }
}
async function logUserParameterChange(username, mode, settings) {
  await ensureParameterHistoryFile(parameterHistoryPath$1);
  const data = await fs.promises.readFile(parameterHistoryPath$1, "utf-8");
  const history = JSON.parse(data);
  const userHistory = history.find((entry) => entry.username === username);
  if (userHistory) {
    const changeDate = (/* @__PURE__ */ new Date()).toISOString();
    userHistory.parameterChanges = userHistory.parameterChanges || [];
    userHistory.parameterChanges.push({ changeDate, mode, settings });
    await fs.promises.writeFile(parameterHistoryPath$1, JSON.stringify(history, null, 2));
  }
}
async function registerUser(username, password, serialNumber) {
  const users = await getUsers(usersFilePath);
  if (users.some((user) => user.username === username)) {
    throw new Error("User already exists");
  }
  if (users.length === 10) {
    throw new Error("Maximum number of users reached");
  }
  const passwordHash = await argon2__namespace.hash(password);
  const newUser = createUser({ username, passwordHash, serialNumber });
  users.push(newUser);
  await saveUser(users);
  await addUserToHistory(username, serialNumber);
}
async function setUser(username, mode, settings) {
  const users = await getUsers(usersFilePath);
  const user = users.find((u) => u.username === username);
  if (!user) {
    throw new Error("User not found");
  }
  user.modes[mode] = settings;
  user.lastUsedMode = mode;
  await saveUser(users);
  await logUserParameterChange(username, mode, settings);
}
async function loginUser(username, password) {
  const users = await getUsers(usersFilePath);
  const user = users.find((u) => u.username === username);
  if (!user) {
    throw new Error("User not found");
  }
  if (!await argon2__namespace.verify(user.passwordHash, password)) {
    throw new Error("Incorrect password");
  }
  await logUserLogin(username);
  return { username, serialNumber: user.serialNumber, lastUsedMode: user.lastUsedMode };
}
async function getSettingsForMode(username, mode) {
  const users = await getUsers(usersFilePath);
  const user = users.find((u) => u.username === username);
  if (!user) {
    throw new Error("User not found");
  }
  return user.modes[mode];
}
const parameterHistoryPath = path.resolve(__dirname, "../../parameterHistory.json");
let pythonProcess = null;
if (utils.is.dev) {
  try {
    require("electron-reload")(__dirname, {
      electron: path.join(__dirname, "../../node_modules/electron")
    });
  } catch (err) {
    console.log("Error setting up electron-reload:", err);
  }
}
const spawnPythonProcess = (pythonPath, scriptPath) => {
  return new Promise((resolve2, reject) => {
    const process2 = child_process.spawn(pythonPath, [scriptPath]);
    process2.stdout.on("data", (data) => {
      console.log(`stdout: ${data}`);
    });
    process2.stderr.on("data", (data) => {
      console.error(`stderr: ${data}`);
    });
    process2.on("close", (code) => {
      console.log(`Python process exited with code ${code}`);
    });
    process2.on("error", (error) => {
      reject(error);
    });
    resolve2(process2);
  });
};
function createWindow() {
  const mainWindow = new electron.BrowserWindow({
    width: 400,
    height: 810,
    resizable: false,
    show: false,
    autoHideMenuBar: true,
    icon: process.platform === "win32" ? iconIco : iconPng,
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
      sandbox: false
    }
  });
  mainWindow.on("ready-to-show", () => {
    mainWindow.show();
  });
  mainWindow.webContents.setWindowOpenHandler((details) => {
    electron.shell.openExternal(details.url);
    return { action: "deny" };
  });
  if (utils.is.dev && process.env["ELECTRON_RENDERER_URL"]) {
    mainWindow.loadURL(process.env["ELECTRON_RENDERER_URL"]);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
}
electron.app.whenReady().then(async () => {
  utils.electronApp.setAppUserModelId("com.electron");
  await ensureUsersFile(usersFilePath);
  const pythonPath = path.resolve(__dirname, "../../src/python/pyEnv/bin/python");
  const scriptPath = path.resolve(__dirname, "../../src/python/mainProcess.py");
  console.log("pythonPath: ", pythonPath);
  electron.app.on("browser-window-created", (_, window) => {
    utils.optimizer.watchWindowShortcuts(window);
  });
  try {
    pythonProcess = await spawnPythonProcess(pythonPath, scriptPath);
  } catch (error) {
    console.error("Error spawning python process: ", error);
  }
  createWindow();
  electron.app.on("activate", function() {
    if (electron.BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    electron.app.quit();
  }
  if (pythonProcess) {
    pythonProcess.kill();
  }
});
electron.ipcMain.handle(
  "register-user",
  async (_, username, password, serialNumber) => {
    try {
      await registerUser(username, password, serialNumber);
      return { success: true };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }
);
electron.ipcMain.handle(
  "set-user",
  async (_, username, mode, settings) => {
    try {
      await setUser(username, mode, settings);
      return { success: true };
    } catch (error) {
      return { success: false, message: error.message };
    }
  }
);
electron.ipcMain.handle("login-user", async (_, username, password) => {
  try {
    const user = await loginUser(username, password);
    return { success: true, user };
  } catch (error) {
    return { success: false, message: error.message };
  }
});
electron.ipcMain.handle("get-settings-for-mode", async (_, username, mode) => {
  try {
    const settings = await getSettingsForMode(username, mode);
    return { success: true, settings };
  } catch (error) {
    return { success: false, message: error.message };
  }
});
electron.ipcMain.handle("download-parameter-log", async (_, username) => {
  try {
    const data = await fs.promises.readFile(parameterHistoryPath, "utf-8");
    const history = JSON.parse(data);
    const userHistory = history.find((entry) => entry.username === username);
    if (!userHistory) {
      throw new Error("User not found");
    }
    const csv = json2csv.parse(userHistory.parameterChanges);
    const directory = electron.app.getPath("downloads");
    const filePath = path__namespace.join(directory, `${username}_parameter_log.csv`);
    await fs.promises.writeFile(filePath, csv);
    return { success: true, directory };
  } catch (error) {
    return { success: false, message: error.message };
  }
});
electron.ipcMain.handle("download-login-history", async (_, username) => {
  try {
    const data = await fs.promises.readFile(parameterHistoryPath, "utf-8");
    const history = JSON.parse(data);
    const userHistory = history.find((entry) => entry.username === username);
    if (!userHistory) {
      throw new Error("User not found");
    }
    const csv = json2csv.parse(userHistory.loginHistory);
    const directory = electron.app.getPath("downloads");
    const filePath = path__namespace.join(directory, `${username}_login_history.csv`);
    await fs.promises.writeFile(filePath, csv);
    return { success: true, directory };
  } catch (error) {
    return { success: false, message: error.message };
  }
});
exports.spawnPythonProcess = spawnPythonProcess;
