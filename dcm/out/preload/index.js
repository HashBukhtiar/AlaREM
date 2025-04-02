"use strict";
const electron = require("electron");
const preload = require("@electron-toolkit/preload");
const api = {
  registerUser: async (username, password, serialNumber) => {
    const result = await electron.ipcRenderer.invoke("register-user", username, password, serialNumber);
    return result;
  },
  setUser: async (username, mode, settings) => {
    const result = await electron.ipcRenderer.invoke("set-user", username, mode, settings);
    return result;
  },
  loginUser: async (username, password) => {
    const result = await electron.ipcRenderer.invoke("login-user", username, password);
    return result;
  },
  getSettingsForMode: async (username, mode) => {
    const result = await electron.ipcRenderer.invoke("get-settings-for-mode", username, mode);
    return result;
  },
  downloadParameterLog: async (username) => {
    const result = await electron.ipcRenderer.invoke("download-parameter-log", username);
    return result;
  },
  downloadLoginHistory: async (username) => {
    const result = await electron.ipcRenderer.invoke("download-login-history", username);
    return result;
  }
};
if (process.contextIsolated) {
  try {
    electron.contextBridge.exposeInMainWorld("electron", preload.electronAPI);
    electron.contextBridge.exposeInMainWorld("api", api);
  } catch (error) {
    console.error(error);
  }
} else {
  window.electron = preload.electronAPI;
  window.api = api;
}
