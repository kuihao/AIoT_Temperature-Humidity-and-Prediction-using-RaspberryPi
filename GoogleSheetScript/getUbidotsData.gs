//Replace the following constants with your Ubidots Token and the Variable(s) you wish to retrieve. 

var TOKEN = "<ubidots token>"
var URL = "http://industrial.api.ubidots.com/api/v1.6/"
var DEVICE_ID = "<ubidots device ID>"
var VARIABLE_TEMPERATURE_ID = "<ubidots variable ID>"
var VARIABLE_HUMIDITY_ID = "<ubidots variable ID>"
var NUMBER_OF_VALUES = "100" // Number of values to retieve from the variable.
var VARIABLE_SHEET = "Newest datas" // Name of the sheet where the variables values will be store.
var SHEET_NAME_HISTORY_TEMPERATURE = "History Temperature" // Name of the sheet where the values from the variable will be store.
var SHEET_NAME_HISTORY_HUMIDITY = "History Humidity"
function onOpen() {
  var sheet = SpreadsheetApp.getActive();

  var menuItems = [
    {name: 'Get newest data', functionName: 'get_newest_data'},
    {name: 'History temperature', functionName: 'get_history_temperature'},
    {name: 'History humidity', functionName: 'get_history_humidity'}
  ];

  sheet.addMenu('Ubidots', menuItems);
}

function get_newest_data(){

  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(VARIABLE_SHEET);

  if (sheet != null) {
    SpreadsheetApp.getActiveSpreadsheet().deleteSheet(sheet);
  }

  sheet = SpreadsheetApp.getActiveSpreadsheet().insertSheet();
  sheet.setName(VARIABLE_SHEET);

  var options =
      {
        "contentType" : "application/json",
        "headers" : {"X-Auth-Token": TOKEN},
        "method": "get"
      };

  var response = UrlFetchApp.fetch(URL + "datasources/"+ DEVICE_ID +"/variables/", options);
  var obj = JSON.parse(response).results

  for (var i = 0, l = obj.length; i < l; i++) {
    var date = new Date(obj[i].last_activity)
    sheet.appendRow([obj[i].name, date, obj[i].last_value.value, obj[i].unit]);
  }
}

function get_history_temperature(){

  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME_HISTORY_TEMPERATURE);
  if (sheet != null) {
    SpreadsheetApp.getActiveSpreadsheet().deleteSheet(sheet);
  }

  sheet = SpreadsheetApp.getActiveSpreadsheet().insertSheet();
  sheet.setName(SHEET_NAME_HISTORY_TEMPERATURE);

  var options =
      {
        "contentType" : "application/json",
        "headers" : {"X-Auth-Token": TOKEN},
        "method": "get"
      };

  var response = UrlFetchApp.fetch( URL + "variables/"+ VARIABLE_TEMPERATURE_ID +"/values/?page_size="+ NUMBER_OF_VALUES, options);
  var obj = JSON.parse(response).results

  for (var i = 0, l = obj.length; i < l; i++) {
    var date = new Date(obj[i].timestamp)
    sheet.appendRow([date, obj[i].value, obj[i].context]);
  }
}

function get_history_humidity(){

  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME_HISTORY_HUMIDITY);
  if (sheet != null) {
    SpreadsheetApp.getActiveSpreadsheet().deleteSheet(sheet);
  }

  sheet = SpreadsheetApp.getActiveSpreadsheet().insertSheet();
  sheet.setName(SHEET_NAME_HISTORY_HUMIDITY);

  var options =
      {
        "contentType" : "application/json",
        "headers" : {"X-Auth-Token": TOKEN},
        "method": "get"
      };

  var response = UrlFetchApp.fetch( URL + "variables/"+ VARIABLE_HUMIDITY_ID +"/values/?page_size="+ NUMBER_OF_VALUES, options);
  var obj = JSON.parse(response).results

  for (var i = 0, l = obj.length; i < l; i++) {
    var date = new Date(obj[i].timestamp)
    sheet.appendRow([date, obj[i].value, obj[i].context]);
  }
}