# https://www.tampermonkey.net/documentation.php?locale=en llms-full.txt

## Tampermonkey Documentation
☰

[Home](https://www.tampermonkey.net/index.php) [Userscripts](https://www.tampermonkey.net/scripts.php) [Support](https://www.tampermonkey.net/faq.php) [Changes](https://www.tampermonkey.net/changelog.php) [Contribute](https://www.tampermonkey.net/contrib.php) [About](https://www.tampermonkey.net/imprint.php)

☰

[FAQ](https://www.tampermonkey.net/faq.php) [Documentation](https://www.tampermonkey.net/documentation.php)

# Table of Contents

Userscript Header

[@name](https://www.tampermonkey.net/documentation.php#meta:name) [@namespace](https://www.tampermonkey.net/documentation.php#meta:namespace) [@copyright](https://www.tampermonkey.net/documentation.php#meta:copyright) [@version](https://www.tampermonkey.net/documentation.php#meta:version) [@description](https://www.tampermonkey.net/documentation.php#meta:description) [@icon, @iconURL, @defaulticon](https://www.tampermonkey.net/documentation.php#meta:icon) [@icon64, @icon64URL](https://www.tampermonkey.net/documentation.php#meta:icon64) [@grant](https://www.tampermonkey.net/documentation.php#meta:grant) [@author](https://www.tampermonkey.net/documentation.php#meta:author) [@homepage, @homepageURL, @website, @source](https://www.tampermonkey.net/documentation.php#meta:homepage) [@antifeature](https://www.tampermonkey.net/documentation.php#meta:antifeature) [@require](https://www.tampermonkey.net/documentation.php#meta:require) [@resource](https://www.tampermonkey.net/documentation.php#meta:resource) [@include](https://www.tampermonkey.net/documentation.php#meta:include) [@match](https://www.tampermonkey.net/documentation.php#meta:match) [@exclude](https://www.tampermonkey.net/documentation.php#meta:exclude) [@run-at](https://www.tampermonkey.net/documentation.php#meta:run_at) [@run-in](https://www.tampermonkey.net/documentation.php#meta:run_in) [@sandbox](https://www.tampermonkey.net/documentation.php#meta:sandbox) [@tag](https://www.tampermonkey.net/documentation.php#meta:tag) [@connect](https://www.tampermonkey.net/documentation.php#meta:connect) [@noframes](https://www.tampermonkey.net/documentation.php#meta:noframes) [@updateURL](https://www.tampermonkey.net/documentation.php#meta:updateURL) [@downloadURL](https://www.tampermonkey.net/documentation.php#meta:downloadURL) [@supportURL](https://www.tampermonkey.net/documentation.php#meta:supportURL) [@webRequest](https://www.tampermonkey.net/documentation.php#meta:webRequest) [@unwrap](https://www.tampermonkey.net/documentation.php#meta:unwrap)

Application Programming Interface

[unsafeWindow](https://www.tampermonkey.net/documentation.php#api:unsafeWindow) [Subresource Integrity](https://www.tampermonkey.net/documentation.php#api:Subresource_Integrity) [GM\_addElement(tag\_name, attributes), GM\_addElement(parent\_node, tag\_name, attributes)](https://www.tampermonkey.net/documentation.php#api:GM_addElement) [GM\_addStyle(css)](https://www.tampermonkey.net/documentation.php#api:GM_addStyle) [GM\_download(details), GM\_download(url, name)](https://www.tampermonkey.net/documentation.php#api:GM_download) [GM\_getResourceText(name)](https://www.tampermonkey.net/documentation.php#api:GM_getResourceText) [GM\_getResourceURL(name)](https://www.tampermonkey.net/documentation.php#api:GM_getResourceURL) [GM\_info](https://www.tampermonkey.net/documentation.php#api:GM_info) [GM\_log(message)](https://www.tampermonkey.net/documentation.php#api:GM_log) [GM\_notification(details, ondone), GM\_notification(text, title, image, onclick)](https://www.tampermonkey.net/documentation.php#api:GM_notification) [GM\_openInTab(url, options), GM\_openInTab(url, loadInBackground)](https://www.tampermonkey.net/documentation.php#api:GM_openInTab) [GM\_registerMenuCommand(name, callback, options\_or\_accessKey)](https://www.tampermonkey.net/documentation.php#api:GM_registerMenuCommand) [GM\_unregisterMenuCommand(menuCmdId)](https://www.tampermonkey.net/documentation.php#api:GM_unregisterMenuCommand) [GM\_setClipboard(data, info, cb)](https://www.tampermonkey.net/documentation.php#api:GM_setClipboard) [GM\_getTab(callback)](https://www.tampermonkey.net/documentation.php#api:GM_getTab) [GM\_saveTab(tab, cb)](https://www.tampermonkey.net/documentation.php#api:GM_saveTab) [GM\_getTabs(callback)](https://www.tampermonkey.net/documentation.php#api:GM_getTabs) [GM\_setValue(key, value)](https://www.tampermonkey.net/documentation.php#api:GM_setValue) [GM\_getValue(key, defaultValue)](https://www.tampermonkey.net/documentation.php#api:GM_getValue) [GM\_deleteValue(key)](https://www.tampermonkey.net/documentation.php#api:GM_deleteValue) [GM\_listValues()](https://www.tampermonkey.net/documentation.php#api:GM_listValues) [GM\_setValues(values)](https://www.tampermonkey.net/documentation.php#api:GM_setValues) [GM\_getValues(keysOrDefaults)](https://www.tampermonkey.net/documentation.php#api:GM_getValues) [GM\_deleteValues(keys)](https://www.tampermonkey.net/documentation.php#api:GM_deleteValues) [GM\_addValueChangeListener(key, (key, old\_value, new\_value, remote) => void)](https://www.tampermonkey.net/documentation.php#api:GM_addValueChangeListener) [GM\_removeValueChangeListener(listenerId)](https://www.tampermonkey.net/documentation.php#api:GM_removeValueChangeListener) [GM\_xmlhttpRequest(details)](https://www.tampermonkey.net/documentation.php#api:GM_xmlhttpRequest) [GM\_webRequest(rules, listener)](https://www.tampermonkey.net/documentation.php#api:GM_webRequest) [GM\_cookie.list(details\[, callback\])](https://www.tampermonkey.net/documentation.php#api:GM_cookie.list) [GM\_cookie.set(details\[, callback\])](https://www.tampermonkey.net/documentation.php#api:GM_cookie.set) [GM\_cookie.delete(details, callback)](https://www.tampermonkey.net/documentation.php#api:GM_cookie.delete) [GM\_audio.setMute(details, callback?)](https://www.tampermonkey.net/documentation.php#api:GM_audio.setMute) [GM\_audio.getState(callback)](https://www.tampermonkey.net/documentation.php#api:GM_audio.getState) [GM\_audio.addStateChangeListener(listener, callback)](https://www.tampermonkey.net/documentation.php#api:GM_audio.addStateChangeListener) [GM\_audio.removeStateChangeListener(listener, callback)](https://www.tampermonkey.net/documentation.php#api:GM_audio.removeStateChangeListener) [window.onurlchange](https://www.tampermonkey.net/documentation.php#api:window.onurlchange) [window.close](https://www.tampermonkey.net/documentation.php#api:window.close) [window.focus](https://www.tampermonkey.net/documentation.php#api:window.focus) [<><!\[CDATA\[...\]\]></>](https://www.tampermonkey.net/documentation.php#api:CDATA)

Settings

[Content Script API](https://www.tampermonkey.net/documentation.php#settings:content_script_api)

# Userscript Header

## @name

The name of the script.

Internationalization is done by adding an appendix naming the locale.

```ts
// @name    A test
// @name:de Ein Test
```

## @namespace

The namespace of the script.

## @copyright

A copyright statement shown at the header of the script's editor right below the script name.

## @version

The script version. This is used for the update check and needs to be increased at every update.

In this list the next entry is considered to be a higher version number, eg: `Alpha-v1` < `Alpha-v2` and `16.4` == `16.04`

- `Alpha-v1`
- `Alpha-v2`
- `Alpha-v10`
- `Beta`
- `0.5pre3`
- `0.5prelimiary`
- `0.6pre4`
- `0.6pre5`
- `0.7pre4`
- `0.7pre10`
- `1.-1`
- `1` == `1.` == `1.0` == `1.0.0`
- `1.1a`
- `1.1aa`
- `1.1ab`
- `1.1b`
- `1.1c`
- `1.1.-1`
- `1.1` == `1.1.0` == `1.1.00`
- `1.1.1.1.1`
- `1.1.1.1.2`
- `1.1.1.1`
- `1.10.0-alpha`
- `1.10` == `1.10.0`
- `1.11.0-0.3.7`
- `1.11.0-alpha`
- `1.11.0-alpha.1`
- `1.11.0-alpha+1`
- `1.12+1` == `1.12+1.0`
- `1.12+1.1` == `1.12+1.1.0`
- `1.12+2`
- `1.12+2.1`
- `1.12+3`
- `1.12+4`
- `1.12`
- `2.0`
- `16.4` == `16.04`
- `2023-08-17.alpha`
- `2023-08-17`
- `2023-08-17_14-04` == `2023-08-17_14-04.0`
- `2023-08-17+alpha`
- `2023-09-11_14-0`

## @description

A short significant description.

Internationalization is done by adding an appendix naming the locale.

```ts
// @description    This userscript does wonderful things
// @description:de Dieses Userscript tut wundervolle Dinge
```

## @icon, @iconURL, @defaulticon

The script icon in low res.

## @icon64, @icon64URL

This scripts icon in 64x64 pixels. If this tag, but `@icon` is given the `@icon` image will be scaled at some places at the options page.

## @grant

`@grant` is used to whitelist `GM_*` and `GM.*` functions, the `unsafeWindow` object and some powerful `window` functions.

```ts
// @grant GM_setValue
// @grant GM_getValue
// @grant GM.setValue
// @grant GM.getValue
// @grant GM_setClipboard
// @grant unsafeWindow
// @grant window.close
// @grant window.focus
// @grant window.onurlchange
```

Since closing and focusing tabs is a powerful feature this needs to be added to the `@grant` statements as well.
In case `@grant` is followed by `none` the sandbox is disabled. In this mode no `GM_*` function but the `GM_info` property will be available.

```ts
// @grant none
```

If no `@grant` tag is given an empty list is assumed. However this different from using `none`.

## @author

The scripts author.

## @homepage, @homepageURL, @website, @source

The authors homepage that is used at the options page to link from the scripts name to the given page. Please note that if the `@namespace` tag starts with `http://` its content will be used for this too.

## @antifeature

This tag allows script developers to disclose whether they monetize their scripts. It is for example required by [GreasyFork](https://greasyfork.org/).

Syntax: <tag> <type> <description>

_<type>_ can have the following values:

- ads
- tracking
- miner

```ts
// @antifeature       ads         We show you ads
// @antifeature:fr    ads         Nous vous montrons des publicités
// @antifeature       tracking    We have some sort of analytics included
// @antifeature       miner       We use your computer's resources to mine a crypto currency
```

Internationalization is done by adding an appendix naming the locale.

## @require

Points to a JavaScript file that is loaded and executed before the script itself starts running.
Note: the scripts loaded via `@require` and their _"use strict"_ statements might influence the userscript's strict mode!

```ts
// @require https://code.jquery.com/jquery-2.1.4.min.js
// @require https://code.jquery.com/jquery-2.1.3.min.js#sha256=23456...
// @require https://code.jquery.com/jquery-2.1.2.min.js#md5=34567...,sha256=6789...
// @require tampermonkey://vendor/jquery.js
// @require tampermonkey://vendor/jszip/jszip.js
```

Please check the [sub-resource integrity](https://www.tampermonkey.net/documentation.php#api:Subresource_Integrity) section for more information how to ensure integrity.

Multiple tag instances are allowed.

## @resource

Preloads resources that can by accessed via `GM_getResourceURL` and `GM_getResourceText` by the script.

```ts
// @resource icon1       http://www.tampermonkey.net/favicon.ico
// @resource icon2       /images/icon.png
// @resource html        http://www.tampermonkey.net/index.html
// @resource xml         http://www.tampermonkey.net/crx/tampermonkey.xml
// @resource SRIsecured1 http://www.tampermonkey.net/favicon.ico#md5=123434...
// @resource SRIsecured2 http://www.tampermonkey.net/favicon.ico#md5=123434...;sha256=234234...
```

Please check the [sub-resource integrity](https://www.tampermonkey.net/documentation.php#api:Subresource_Integrity) section for more information how to ensure integrity.

Multiple tag instances are allowed.

## @include

The pages on that a script should run. Multiple tag instances are allowed.
@include doesn't support the URL hash parameter. You have to match the path without the hash parameter and make use of [window.onurlchange](https://www.tampermonkey.net/documentation.php#api:window.onurlchange)

```ts
// @include http://www.tampermonkey.net/*
// @include http://*
// @include https://*
// @include /^https:\/\/www\.tampermonkey\.net\/.*$/
// @include *
```

Note: When writing something like `*://tmnk.net/*` many script developers expect the script to run at `tmnk.net` only, but this is not the case.
It also runs at `https://example.com/?http://tmnk.net/` as well.

Therefore Tampermonkey interprets `@includes` that contain a `://` a little bit like `@match`. Every `*` before `://` only matches everything except `:` characters to makes sure only the URL scheme is matched.
Also, if such an `@include` contains a `/` after `://`, then everything between those strings is treat as host, matching everything except `/` characters. The same applies to `*` directly following `://`.

## @match

In Tampermonkey, the `@match` directive is used to specify the web pages that your script should run on.
The value of `@match` should be a URL pattern that matches the pages you want your script to run on. Here are the parts of the URL pattern that you'll need to set:

```ts
// @match <protocol>://<domain><path>
```

- **protocol** \- This is the first part of the URL, before the colon. It specifies the protocol that the page uses, such as `http` or `https`. `*` matches both.
- **domain** \- This is the second part of the URL, after the protocol and two slashes. It specifies the domain name of the website, such as `tmnk.com`. You can use the wildcard character this way `*.tmnk.net` to match `tmnk.net` and any sub-domain of it like `www.tmnk.net`.
- **path** \- This is the part of the URL that comes after the domain name, and may include additional subdirectories or filenames. You can use the wildcard character `*` to match any part of the path.

Please check [this documentation](https://developer.chrome.com/docs/extensions/mv2/match_patterns/) to get more information about match pattern. Note: the `<all_urls>` statement is not yet supported and the scheme part also accepts `http*://`.

Multiple tag instances are allowed.

More examples:

```ts*
// @match *://*/*
// @match https://*/*
// @match http://*/foo*
// @match https://*.tampermonkey.net/foo*bar
```

## @exclude

Exclude URLs even it they are included by `@include` or `@match`.

Multiple tag instances are allowed.

## @run-at

Defines the moment the script is injected.
In opposition to other script handlers, `@run-at` defines the first possible moment a script wants to run.
This means it may happen, that a script that uses the `@require` tag may be executed after the document is already loaded, cause fetching the required script took that long.

Anyhow, all `DOMNodeInserted`, `DOMContentLoaded` and `load` events fired after the given injection moment are cached and delivered to listeners registered via the sandbox's `window.addEventListener` method.

```ts
// @run-at document-start
```

The script will be injected as fast as possible.

```ts
// @run-at document-body
```

The script will be injected if the body element exists.

```ts
// @run-at document-end
```

The script will be injected when or after the DOMContentLoaded event was dispatched.

```ts
// @run-at document-idle
```

The script will be injected after the DOMContentLoaded event was dispatched. This is the default value if no `@run-at` tag is given.

```ts
// @run-at context-menu
```

The script will be injected if it is clicked at the browser context menu.

Note: all `@include` and `@exclude` statements will be ignored if this value is used, but this may change in the future.

## @run-in v5.3+

Defines the type of browser context in which the script is injected. This meta key allows you to control whether the script should run in normal browsing tabs, incognito tabs, or both. This provides flexibility in determining the script's behavior based on the privacy context of the browsing session.

```ts
// @run-in normal-tabs
```

The script will be injected only in normal browsing tabs (non-incognito mode, default container).

```ts
// @run-in incognito-tabs
```

The script will be injected only in incognito browsing tabs (private mode). In Firefox, this means all tabs that don't use the default cookie store.

Firefox supports containers, which allow you to separate your browsing activities into distinct contexts. You can specify the container ID in the `@run-in` tag to control the script's behavior based on the container context.

```ts
// @run-in container-id-2
// @run-in container-id-3
```

The script will be injected only in tabs that belong to the specified containers. The container ID can be found by checking `GM_info.container` when the script is running in the desired container context.

If no `@run-in` tag is specified, the script defaults to being injected in all tabs.

## @sandbox 4.18+

`@sandbox` allows Tampermonkey to decide where the userscript is injected:

- `MAIN_WORLD` \- the page
- `ISOLATED_WORLD` \- the extension's content script
- `USERSCRIPT_WORLD` \- a special context created for userscripts

But instead of specifying an environment, the userscript can express what exactly it needs access to. `@sandbox` supports three possible arguments:

- `raw`
"Raw" access means that a script for compatibility reasons always needs to run in page context, the `MAIN_WORLD`.
At the moment this mode is the default if `@sandbox` is omitted.
If injection into the `MAIN_WORLD` is not possible (e.g. because of a CSP) the userscript will be injected into other (enabled) sandboxes according to the order of this list.

- `JavaScript`
"JavaScript" access mode means that this script needs access to `unsafeWindow`.
At Firefox a special context, the `USERSCRIPT_WORLD`, is created which also bypasses existing CSPs. It however, might create new issues since now [`cloneInto` and `exportFunction`](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Sharing_objects_with_page_scripts) are necessary to share objects with the page.
`raw` mode is used as fallback at other browsers.

- `DOM`
Use this access mode if the script only needs DOM and no direct `unsafeWindow` access.
If [enabled](https://www.tampermonkey.net/faq#Q404) these scripts are executed inside the extension context, the `ISOLATED_WORLD`, or at any other enabled context otherwise, because they all grant DOM access.


```ts
// @sandbox JavaScript
```

## @tag

You can add tags to your script which will be visible in the script list if this tag is part of your system's tag list.
Tags can be useful to categorize your scripts or to mark them as a certain type.
The list of tags can be found at the script's settings page.

Example of a script with tags

```ts
// ==UserScript==
// @name         My Script
// @tag          productivity
// ==/UserScript==
```

## @connect

This tag defines the domains (no top-level domains) including subdomains which are allowed to be retrieved by [GM\_xmlhttpRequest](https://www.tampermonkey.net/documentation.php#api:GM_xmlhttpRequest)

```ts
// @connect <value>
```

`<value>` can be:

- a domain name like `example.com` (this will also allow all subdomains).
- a subdomain name like `subdomain.example.com`.
- `self` to whitelist the domain the script is currently running at.
- `localhost` to access the localhost.
- an IP address like `1.2.3.4`.
- `*`.

If it's not possible to declare all domains a userscript might connect to then it's a good practice to do the following:

1. Declare all known or at least all common domains that might be connected by the script to avoid the confirmation dialog for most users.
2. Additionally add `@connect *` to the script to allow Tampermonkey to offer an "Always allow all domains" button.

Users can also whitelist all requests by adding `*` to the user domain whitelist at the script settings tab.

Notes:

- Both, the initial **and** the final URL will be checked!
- For backward compatibility to Scriptish [`@domain`](https://github.com/scriptish/scriptish/wiki/Manual%3A-Metadata-Block#user-content-domain-new-in-scriptish) tags are interpreted as well.
- Multiple tag instances are allowed.

More examples:

```ts
// @connect tmnk.net
// @connect www.tampermonkey.net
// @connect self
// @connect localhost
// @connect 8.8.8.8
// @connect *
```

## @noframes

This tag makes the script running on the main pages, but not at iframes.

## @updateURL

An update URL for the userscript.
Note: a `@version` tag is required to make update checks work.

## @downloadURL

Defines the URL where the script will be downloaded from when an update was detected. If the value _none_ is used, then no update check will be done.

## @supportURL

Defines the URL where the user can report issues and get personal support.

## @webRequest

Note: this API is experimental and might change at any time. It is also not available anymore at Manifest v3 versions of Tampermonkey 5.2+ (Chrome and derivates).

`@webRequest` takes a JSON document that matches [`GM_webRequest`](https://www.tampermonkey.net/documentation.php#api:GM_webRequest)'s `rule` parameter. It allows the rules to apply even before the userscript is loaded.

## @unwrap

Injects the userscript without any wrapper and sandbox into the page, which might be useful for Scriptlets.

# Application Programming Interface

## unsafeWindow

The `unsafeWindow` object provides access to the `window` object of the page that Tampermonkey is running on, rather than the `window` object of the Tampermonkey extension. This can be useful in some cases, such as when a userscript needs to access a JavaScript library or variable that is defined on the page.

## Subresource Integrity

Subresource Integrity (SRI) is a security feature that allows userscript developers to ensure that the external resources (such as JavaScript libraries and CSS files) that they include in their userscript have not been tampered with or modified. This is accomplished by generating a cryptographic hash of the resource and including it in `@require` and `@resource` tags. When the userscript is installed, Tampermonkey will calculate the hash of the resource and compare it to the included hash. If the two hashes do not match, Tampermonkey will refuse to load the resource, preventing attackers from injecting malicious code into your userscript.

The hash component of the URL of `@resource` and `@require` tags is used for this purpose.

```ts
// @resource SRIsecured1 http://example.com/favicon1.ico#md5=ad34bb...
// @resource SRIsecured2 http://example.com/favicon2.ico#md5=ac3434...,sha256=23fd34...
// @require              https://code.jquery.com/jquery-2.1.1.min.js#md5=45eef...
// @require              https://code.jquery.com/jquery-2.1.2.min.js#md5-ac56d...,sha256-6e789...
// @require              https://code.jquery.com/jquery-3.6.0.min.js#sha256-/xUj+3OJU...ogEvDej/m4=
```

Tampermonkey supports `SHA-256` and `MD5` hashes natively, all other (`SHA-1`, `SHA-384` and `SHA-512`) depend on [window.crypto](https://developer.mozilla.org/en-US/docs/Web/API/Crypto).

In case multiple hashes (separated by comma or semicolon) are given the last currently supported one is used by Tampermonkey. All hashes need to be encoded in either hex or Base64 format.

## GM\_addElement(tag\_name, attributes), GM\_addElement(parent\_node, tag\_name, attributes)

`GM_addElement` allows Tampermonkey scripts to add new elements to the page that Tampermonkey is running on. This can be useful for a variety of purposes, such as adding `script` and `img` tags if the page limits these elements with a content security policy (CSP).

It creates an HTML element specified by _"tag\_name"_ and applies all given _"attributes"_ and returns the injected HTML element. If a _"parent\_node"_ is given, then it is attached to it or to document head or body otherwise.

For suitable _"attributes"_, please consult the appropriate documentation. For example:

- [script tag](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script)
- [img tag](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img)
- [style tag](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/style)

```ts
GM_addElement('script', {
  textContent: 'window.foo = "bar";'
});

GM_addElement('script', {
  src: 'https://example.com/script.js',
  type: 'text/javascript'
});

GM_addElement(document.getElementsByTagName('div')[0], 'img', {
  src: 'https://example.com/image.png'
});

GM_addElement(shadowDOM, 'style', {
  textContent: 'div { color: black; };'
});
```

Note: this feature is experimental and the API may change.

## GM\_addStyle(css)

Adds the given style to the document and returns the injected style element.

## GM\_download(details), GM\_download(url, name)

`GM_download` allows userscripts to download a file from a specified URL and save it to the user's local machine.

The `GM_download` function takes the following parameters:

_details_ can have the following attributes:

- **url**: The URL of the file to download or a `Blob` or `File` objectv5.4.6226+. In case of a string, this must be a valid URL and must point to a file that is accessible to the user.
- **name**:
The name to use for the downloaded file.
This should include the file's extension, such as .txt or .pdf.
For security reasons the file extension needs to be whitelisted at Tampermonkey's options page
- **headers**:
An object containing HTTP headers to include in the download request.
See [`GM_xmlhttpRequest`](https://www.tampermonkey.net/documentation.php#meta:GM_xmlhttpRequest) for more details.
- **saveAs**:
A boolean value indicating whether to use the user's default download location, or to prompt the user to choose a different location.
This option works in browser API mode only.
- **conflictAction**:
A string that control what happens when a file with this name already exists. This option works in browser API mode only.
Possible values are `uniquify`, `overwrite` and `prompt`.
Please check [this link](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/downloads/FilenameConflictAction) for more details.
- **onload**: A function to call when the download has completed successfully.
- **onerror**: A function to call if the download fails or is cancelled.
- **onprogress** A callback to be executed if this download made some progress.
- **ontimeout** A callback to be executed if this download failed due to a timeout.

The _download_ argument of the _onerror_ callback can have the following attributes:

- **error**: error reason
  - not\_enabled - the download feature isn't enabled by the user
  - not\_whitelisted - the requested file extension is not whitelisted
  - not\_permitted - the user enabled the download feature, but did not give the _downloads_ permission
  - not\_supported - the download feature isn't supported by the browser/version
  - not\_succeeded - the download wasn't started or failed, the _details_ attribute may provide more information
- **details**: detail about that error

Returns an object with the following property:

- **abort**: A function which can be called to cancel this download.

If `GM.download` is used it returns a promise that resolves to the download details and also has an `abort` function.

Depending on the download mode `GM_info` provides a property called `downloadMode` which is set to one of the following values: **native**, **disabled** or **browser**.

```ts
GM_download("http://example.com/file.txt", "file.txt");

const download = GM_download({
    url: "http://example.com/file.txt",
    name: "file.txt",
    saveAs: true
});

// cancel download after 5 seconds
window.setTimeout(() => download.abort(), 5000);
```

Note: The browser might modify the desired filename. Especially a file extension might be added if the browser finds this to be safe to download at the current OS.

## GM\_getResourceText(name)

Allows userscripts to access the text of a resource (such as a JavaScript or CSS file) that has been included in a userscript via `@resource`.

The function takes a single parameter, which is the _"name"_ of the resource to retrieve. It returns the text of the resource as a string.

Here is an example of how the function might be used:

```ts
const scriptText = GM_getResourceText("myscript.js");
const scriptText2 = await GM.getResourceText("myscript.js");
const script = document.createElement("script");
script.textContent = scriptText;
document.body.appendChild(script);
```

## GM\_getResourceURL(name)

`GM_getResourceURL` allows userscripts to access the URL of a resource (such as a CSS or image file) that has been included in the userscript via a `@resource` tag at the script header.

The function takes a single parameter, which is the _"name"_ of the resource to retrieve. It returns the URL of the resource as a string.

```ts
const imageUrl = GM_getResourceURL("myimage.png");
const imageUrl2 = await GM.getResourceUrl("myimage.png");
const image = document.createElement("img");
image.src = imageUrl;
document.body.appendChild(image);
```

**Important:**: The promise-based version of this function is called `GM.getResourceUrl` (with a lowercase "r" and "l" in "Url").

## GM\_info

Get some info about the script and TM. The object might look like this:

```ts
type ScriptGetInfo = {
    container?: { // 5.3+ | Firefox only
        id: string,
        name?: string
    },
    downloadMode: string,
    isFirstPartyIsolation?: boolean,
    isIncognito: boolean,
    sandboxMode: SandboxMode, // 4.18+
    scriptHandler: string,
    scriptMetaStr: string | null,
    scriptUpdateURL: string | null,
    scriptWillUpdate: boolean,
    userAgentData: UADataValues, // 4.19+
    version?: string,
    script: {
        antifeatures: { [antifeature: string]: { [locale: string]: string } },
        author: string | null,
        blockers: string[],
        connects: string[],
        copyright: string | null,
        deleted?: number | undefined,
        description_i18n: { [locale: string]: string } | null,
        description: string,
        downloadURL: string | null,
        excludes: string[],
        fileURL: string | null,
        grant: string[],
        header: string | null,
        homepage: string | null,
        icon: string | null,
        icon64: string | null,
        includes: string[],
        lastModified: number,
        matches: string[],
        name_i18n: { [locale: string]: string } | null,
        name: string,
        namespace: string | null,
        position: number,
        resources: Resource[],
        supportURL: string | null,
        system?: boolean | undefined,
        'run-at': string | null,
        'run-in': string[] | null, // 5.3+
        unwrap: boolean | null,
        updateURL: string | null,
        version: string,
        webRequest: WebRequestRule[] | null,
        options: {
            check_for_updates: boolean,
            comment: string | null,
            compatopts_for_requires: boolean,
            compat_wrappedjsobject: boolean,
            compat_metadata: boolean,
            compat_foreach: boolean,
            compat_powerful_this: boolean | null,
            sandbox: string | null,
            noframes: boolean | null,
            unwrap: boolean | null,
            run_at: string | null,
            run_in: string | null, // 5.3+
            override: {
                use_includes: string[],
                orig_includes: string[],
                merge_includes: boolean,
                use_matches: string[],
                orig_matches: string[],
                merge_matches: boolean,
                use_excludes: string[],
                orig_excludes: string[],
                merge_excludes: boolean,
                use_connects: string[],
                orig_connects: string[],
                merge_connects: boolean,
                use_blockers: string[],
                orig_run_at: string | null,
                orig_run_in: string[] | null, // 5.3+
                orig_noframes: boolean | null
            }
        }
    }
};

type SandboxMode = 'js' | 'raw' | 'dom';

type Resource = {
    name: string,
    url: string,
    error?: string,
    content?: string,
    meta?: string
};

type WebRequestRule = {
    selector: { include?: string | string[], match?: string | string[], exclude?: string | string[] } | string,
    action: string | {
        cancel?: boolean,
        redirect?: {
            url: string,
            from?: string,
            to?: string
        } | string
    }
};

type UADataValues = {
    brands?: {
        brand: string;
        version: string;
    }[],
    mobile?: boolean,
    platform?: string,
    architecture?: string,
    bitness?: string
}
```

## GM\_log(message)

Log a message to the console.

## GM\_notification(details, ondone), GM\_notification(text, title, image, onclick)

`GM_notification` allows users to display notifications on the screen, using a provided message and other optional parameters.

The function takes several parameters. Either a _details_ object or multiple parameters.

The _details_ object can have the following attributes, from which some can also be used as direct parameter.

The available options include:

- **text**: A string containing the message to display in the notification.
- **title**: The title of the notification.
- **tag**: v5.0+ This tag will be used to identify this notification. This way you can update existing notifications by calling `GM_notification` again and using the same tag. If you don't provide a tag, a new notification will be created every time.
- **image**: The URL of an image to display in the notification.
- **highlight**: A boolean flag whether to highlight the tab that sends the notfication (required unless text is set)
- **silent**: A boolean flag whether to not play a sound
- **timeout**: The time, in milliseconds, after which the notification should automatically close.
- **url**: v5.0+ A URL to load when the user clicks on the notification. You can prevent loading the URL by calling `event.preventDefault()` in the `onclick` event handler.
- **onclick**: A callback function that will be called when the user clicks on the notification.
- **ondone** A callback function that will be called when the notification is closed (no matter if this was triggered by a timeout or a click) or the tab was highlighted

The function does not return a value.

If no `url` and no `tag` is provided the notification will closed when the userscript unloads v5.0+(e.g. when the page is reloaded or the tab is closed).

Here is an example of how the function might be used:

```ts
GM_notification({
  text: "This is the notification message.",
  title: "Notification Title",
  url: 'https:/example.com/',
  onclick: (event) => {
    // The userscript is still running, so don't open example.com
    event.preventDefault();
    // Display an alert message instead
    alert('I was clicked!')
  }
});

const clicked = await GM.notification({ text: "Click me." });
```

## GM\_openInTab(url, options), GM\_openInTab(url, loadInBackground)

`GM_openInTab` allows userscripts to open a new tab in the browser and navigate to a specified URL.

The function takes two parameters:

A string names _"url"_ containing the URL of the page to open in the new tab.

An optional options object that can be used to customize the behavior of the new tab. The available options include:

- **active**: A boolean value indicating whether the new tab should be active (selected) or not. The default is false.
- **insert**: An integer indicating the position at which the new tab should be inserted in the tab strip. The default is false, which means the new tab will be added to the end of the tab strip.
- **setParent**: A boolean value indicating whether the new tab should be considered a child of the current tab. The default is false.
- **incognito** A boolean value that makes the tab being opened inside a incognito mode/private mode window.
- **loadInBackground** A boolean value has the opposite meaning of **active** and was added to achieve Greasemonkey 3.x compatibility.

The function returns an object with the function **close**, the listener **onclose** and a flag called **closed**.

Here is an example of how the function might be used:

```ts
// Open a new tab and navigate to the specified URL
GM_openInTab("https://www.example.com/");
```

## GM\_registerMenuCommand(name, callback, options\_or\_accessKey)

`GM_registerMenuCommand` allows userscripts to add a new entry to the userscript's menu in the browser, and specify a function to be called when the menu item is selected.
Menu items created from different frames are merged into a single menu entry if name, title and accessKey are the same.

The function takes three parameters:

- **name** \- _string_, A string containing the text to display for the menu item.
- **callback** \- _function_, A function to be called when the menu item is selected. The function will be passed a single parameter, which is the currently active tab. As of Tampermonkey 4.14 a MouseEvent or KeyboardEvent is passed as function argument.
- **accessKey** \- _string?_, An optional access key. Please see the description below. Either `options` or `accessKey` can be specified.
- **options** v4.20+ _object?_, Optional options that can be used to customize the menu item. The options are specified as an object with the following properties:
  - **id** v5.0+ _number\|string?_, An optional number that was returned by a previous `GM_registerMenuCommand` call. If specified, the according menu item will be updated with the new options. If not specified or the menu item can't be found, a new menu item will be created.
  - **accessKey** \- _string?_, An optional access key for the menu item. This can be used to create a shortcut for the menu item. For example, if the access key is "s", the user can select the menu item by pressing "s" when Tampermonkey's popup-menu is open. Please note that there are browser-wide shortcuts configurable to open Tampermonkey's popup-menu. (`chrome://extensions/shortcuts` in Chrome, `about:addons` \+ "Manage Extension Shortcuts" in Firefox)
  - **autoClose** \- _boolean?_, An optional boolean parameter that specifies whether the popup menu should be closed after the menu item is clicked. The default value is `true`. Please note that this setting has no effect on the menu command section that is added to the page's context menu.
  - **title** v5.0+ \- _string?_, An optional string that specifies the title of the menu item. This is displayed as a tooltip when the user hovers the mouse over the menu item.

The function return a menu entry ID that can be used to unregister the command.

Here is an example of how the function might be used:

```ts
const menu_command_id_1 = GM_registerMenuCommand("Show Alert", function(event: MouseEvent | KeyboardEvent) {
  alert("Menu item selected");
}, {
  accessKey: "a",
  autoClose: true
});

const menu_command_id_2 = GM_registerMenuCommand("Log", function(event: MouseEvent | KeyboardEvent) {
  console.log("Menu item selected");
}, "l");
```

## GM\_unregisterMenuCommand(menuCmdId)

`GM_unregisterMenuCommand` removes an existing entry from the userscript's menu in the browser.

The function takes a single parameter, which is the ID of the menu item to remove. It does not return a value.

Here is an example of how the function might be used:

```ts
const menu_command_id = GM_registerMenuCommand(...);
GM_unregisterMenuCommand(menu_command_id);
```

## GM\_setClipboard(data, info, cb)

`GM_setClipboard` sets the text of the clipboard to a specified value.

The function takes a parameter _"data"_, which is the string to set as the clipboard text, a parameter _"info"_ and an optional callback function _"cb"_.

_"info_" can be just a string expressing the type `text` or `html` or an object like
_"cb"_ is an optional callback function that is called when the clipboard has been set.

```ts
{
    type: 'text',
    mimetype: 'text/plain'
}
```

```ts
GM_setClipboard("This is the clipboard text.", "text", () => console.log("Clipboard set!"));
await GM.setClipboard("This is the newer clipboard text.", "text");
console.log('Clipboard set again!');
```

## GM\_getTab(callback)

The GM\_getTab function takes a single parameter, a callback function that will be called with an object that is persistent as long as this tab is open.

```ts
GM_getTab((tab) => console.log(tab));
const t = await GM.getTab();
console.log(t);
```

## GM\_saveTab(tab, cb)

The `GM_saveTab` function allows a userscript to save information about a tab for later use.

The function takes a _"tab_" parameter, which is an object containing the information to be saved about the tab and an optional callback function _"cb"_.

The `GM_saveTab` function saves the provided tab information, so that it can be retrieved later using the `GM_getTab` function.

Here is an example of how the GM\_saveTab function might be used in a userscript:

```ts
GM_getTab(function(tab) {
    tab.newInfo = "new!";
    GM_saveTab(tab);
});
const tab = await GM.getTab();
await GM.saveTab(tab);
```

In this example, the `GM_saveTab` function is called with the tab object returned by the `GM_getTab` function, and a new key called "newInfo".

## GM\_getTabs(callback)

The `GM_getTabs` function takes a single parameter: a callback function that will be called with the information about the tabs.

The _"tabs"_ object that is passed to the callback function contains objects, with each object representing the saved tab information stored by `GM_saveTab`.

```ts
GM_getTabs((tabs) => {
    for (const [tabId, tab] of Object.entries(tabs)) {
        console.log(`tab ${tabId}`, tab);
    }
});
const tabs = await GM.getTabs();
```

## GM\_setValue(key, value)

The `GM_setValue` allows a userscript to set the value of a specific key in the userscript's storage.

The `GM_setValue` function takes two parameters:

- A string specifying the key for which the value should be set.
- The value to be set for the key. Values (including nested object properties) can be `null` or of type "object", "string", "number", "undefined" or "boolean".

The `GM_setValue` function does not return any value. Instead, it sets the provided value for the specified key in the userscript's storage.

Here is an example of how `GM_setValue` and its async pendant `GM.setValue` might be used in a userscript:

```ts
GM_setValue("someKey", "someData");
await GM.setValue("otherKey", "otherData");
```

## GM\_getValue(key, defaultValue)

The `GM_getValue` function allows a userscript to retrieve the value of a specific key in the userscript's storage.
It takes two parameters:

- A string specifying the key for which the value should be retrieved.
- A default value to be returned if the key does not exist in the userscript's storage. This default value can be of any type (string, number, object, etc.).

The `GM_getValue` function returns the value of the specified key from the userscript's storage, or the default value if the key does not exist.

Here is an example of how the `GM_getValue` function might be used in a userscript:

```ts
const someKey = GM_getValue("someKey", null);
const otherKey = await GM.getValue("otherKey", null);
```

In this example, the `GM_getValue` function is called with the key "someKey" and a default value of null.
If the "someKey" key exists in the userscript's storage, its value will be returned and stored in the someKey variable.
If the key does not exist, the default value of null will be returned and stored in the savedTab variable.

## GM\_deleteValue(key)

Deletes _"key"_ from the userscript's storage.

```ts
GM_deleteValue("someKey");
await GM.deleteValue("otherKey");
```

## GM\_listValues()

The `GM_listValues` function returns a list of keys of all stored data.

```ts
const keys = GM_listValues();
const asyncKeys = await GM.listValues();
```

## GM\_setValues(values) v5.3+

The `GM_setValues` function allows a userscript to set multiple key-value pairs in the userscript's storage simultaneously.

The `GM_setValues` function takes one parameter:

- An object where each key-value pair corresponds to a key and the value to be set for that key. Values (including nested object properties) can be `null` or of type "object", "string", "number", "undefined" or "boolean".

The `GM_setValues` function does not return any value. Instead, it sets the provided values for the specified keys in the userscript's storage.

Here is an example of how `GM_setValues` and its async counterpart `GM.setValues` might be used in a userscript:

```ts
GM_setValues({ foo: 1, bar: 2 });
await GM.setValues({ foo: 1, bar: 2 });
```

## GM\_getValues(keysOrDefaults) v5.3+

The `GM_getValues` function allows a userscript to retrieve the values of multiple keys in the userscript's storage. It can also provide default values if the keys do not exist.

The `GM_getValues` function takes one parameter:

- Either an array of strings specifying the keys for which the values should be retrieved, or an object specifying the default values to be returned if the keys do not exist. This default values object can contain keys of any type (string, number, object, etc.).

The `GM_getValues` function returns an object containing the values of the specified keys from the userscript's storage, or the default values if the keys do not exist.

Here is an example of how the `GM_getValues` function might be used in a userscript:

```ts
const values = GM_getValues(['foo', 'bar']);
const asyncValues = await GM.getValues(['foo', 'bar']);

const defaultValues = GM_getValues({ foo: 1, bar: 2, baz: 3 });
const asyncDefaultValues = await GM.getValues({ foo: 1, bar: 2, baz: 3 });
```

In this example, the `GM_getValues` function is called with an array of keys or an object of default values. It returns an object with the values of the specified keys or the default values if the keys do not exist.

## GM\_deleteValues(keys) v5.3+

The `GM_deleteValues` function allows a userscript to delete multiple keys from the userscript's storage simultaneously.

The `GM_deleteValues` function takes one parameter:

- An array of strings specifying the keys to be deleted from the userscript's storage.

The `GM_deleteValues` function does not return any value. Instead, it deletes the specified keys from the userscript's storage.

Here is an example of how `GM_deleteValues` and its async counterpart GM.deleteValues might be used in a userscript:

```ts
GM_deleteValues(['foo', 'bar']);
await GM.deleteValues(['foo', 'bar']);
```

## GM\_addValueChangeListener(key, (key, old\_value, new\_value, remote) => void)

The `GM_addValueChangeListener` function allows a userscript to add a listener for changes to the value of a specific key in the userscript's storage.

The function takes two parameters:

- A string specifying the key for which changes should be monitored.
- A callback function that will be called when the value of the key changes. The callback function should have the following signature:

```ts
    function(key, oldValue, newValue, remote) {
        // key is the key whose value has changed
        // oldValue is the previous value of the key
        // newValue is the new value of the key
        // remote is a boolean indicating whether the change originated from a different userscript instance
    }
```


The `GM_addValueChangeListener` function returns a _"listenerId"_ value that can be used to remove the listener later using the `GM_removeValueChangeListener` function.
The very same applies to `GM.addValueChangeListener` and `GM.removeValueChangeListener` with the only difference that both return a promise;

Here is an example of how the `GM_addValueChangeListener` function might be used in a userscript:

```ts
// Add a listener for changes to the "savedTab" key
var listenerId = GM_addValueChangeListener("savedTab", function(key, oldValue, newValue, remote) {
  // Print a message to the console when the value of the "savedTab" key changes
  console.log("The value of the '" + key + "' key has changed from '" + oldValue + "' to '" + newValue + "'");
});
```

`GM_addValueChangeListener` can be used by userscripts to communicate with other userscript instances at other tabs.

## GM\_removeValueChangeListener(listenerId)

`GM_removeValueChangeListener` and `GM.removeValueChangeListener` both get one argument called _"listenerId"_ and remove the change listener with this ID.

## GM\_xmlhttpRequest(details)

The `GM_xmlhttpRequest` allows a userscripts to send an HTTP request and handle the response.
The function takes a single parameter: an object containing the details of the request to be sent and the callback functions to be called when the response is received.

The object can have the following properties:

- **method** \- _string_, usually one of GET, HEAD, POST, PUT, DELETE, ...
- **url** \- _string\|URL\|File\|Blob_, the destination URL or a `Blob` or `File` objectv5.4.6226+
- **headers** e.g. `user-agent`, `referer`, ...
(some special headers are not supported by Safari and Android browsers)
- **data** \- _string\|Blob\|File\|Object\|Array\|FormData\|URLSearchParams?_, some data to send via a POST request
- **redirect** one of `follow`, `error` or `manual`; controls what to happen when a redirect is detected (build 6180+, enforces `fetch` mode)
- **cookie** a cookie to be patched into the sent cookie set
- **cookiePartition** v5.2+ _object_?, containing the partition key to be used for sent and received [partitioned cookies](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies#storage_partitioning)
  - **topLevelSite** _string_?, representing the top frame site for partitioned cookies
- **binary** send the data string in binary mode
- **nocache** don't cache the resource
- **revalidate** revalidate maybe cached content
- **timeout** a timeout in ms
- **context** a property which will be added to the response object
- **responseType** one of `arraybuffer`, `blob`, `json` or `stream`
- **overrideMimeType** a MIME type for the request
- **anonymous** don't send cookies with the request (enforces `fetch` mode)
- **fetch** use a `fetch` instead of a `XMLHttpRequest` request
(at Chrome this causes `details.timeout` and `xhr.onprogress` to not work and makes `xhr.onreadystatechange` receive only `readyState``DONE` (==4) events)
- **user** a user name for authentication
- **password** a password
- **onabort** callback to be executed if the request was aborted
- **onerror** callback to be executed if the request ended up with an error
- **onloadstart** callback to be executed on load start, provides access to the stream object if responseType is set to `stream`
- **onprogress** callback to be executed if the request made some progress
- **onreadystatechange** callback to be executed if the request's `readyState` changed
- **ontimeout** callback to be executed if the request failed due to a timeout
- **onload** callback to be executed if the request was loaded.

```ts
    function(response) {
      // response is an object containing the details of the response
    }
```

**response** has the following attributes:
  - **finalUrl** \- the final URL after all redirects from where the data was loaded
  - **readyState** \- the request's `readyState`
  - **status** \- the request's status
  - **statusText** \- the request's status text
  - **responseHeaders** \- the request's response headers
  - **response** \- the response data as object if `details.responseType` was set
  - **responseXML** \- the response data as XML document
  - **responseText** \- the response data as plain string

`GM_xmlhttpRequest` returns an object with the following property:

- **abort** \- function to be called to cancel this request

`GM.xmlHttpRequest` returns a promise that resolves to the response and also has an `abort` function.

Here is an example of how the `GM_xmlhttpRequest` function might be used in a userscript:

```ts
GM_xmlhttpRequest({
  method: "GET",
  url: "https://example.com/",
  headers: {
    "Content-Type": "application/json"
  },
  onload: function(response) {
    console.log(response.responseText);
  }
});

const r = await GM.xmlHttpRequest({ url: "https://example.com/" }).catch(e => console.error(e));
console.log(r.responseText);
```

**Note:** the `synchronous` flag at `details` is not supported

**Important:**:

- If you want to use this method then please also check the documentation about [`@connect`](https://www.tampermonkey.net/documentation.php#meta:connect)
- The promise-based version of this function is called `GM.xmlHttpRequest` (with a uppercase "h" in "http")

## GM\_webRequest(rules, listener)

Note: this API is experimental and might change at any time. It is also not available anymore at Manifest v3 versions of Tampermonkey 5.2+ (Chrome and derivates).

`GM_webRequest` (re-)registers rules for web request manipulations and the listener of triggered rules.
If you need to just register rules it's better to use `@webRequest` header.
Note, webRequest proceeds only requests with types `sub_frame`, `script`, `xhr` and `websocket`.

### Parameters:

- **rules** \- _object\[\]_, array of rules with following properties:
  - **selector** \- _string\|object_, for which URLs the rule should be triggered, string value is shortening for `{ include: [selector] }`, object properties:
    - **include** \- _string\|string\[\]_, URLs, patterns, and regexpes for rule triggering;
    - **match** \- _string\|string\[\]_, URLs and patterns for rule trigering;
    - **exclude** \- _string\|string\[\]_, URLs, patterns, and regexpes for not triggering the rule;
  - **action** \- _string\|object_, what to do with the request, string value `"cancel"` is shortening for `{ cancel: true }`, object properties:
    - **cancel** \- _boolean_, whether to cancel the request;
    - **redirect** \- _string\|object_, redirect to some URL which must be included in any @match or @include header. When a string, redirects to the static URL. If object:
      - **from** \- _string_, a regexp to extract some parts of the URL, e.g. `"([^:]+)://match.me/(.*)"`;
      - **to** \- _string_, pattern for substitution, e.g. `"$1://redirected.to/$2"`;
- **listener** \- _function_, is called when the rule is triggered, cannot impact on the rule action, arguments:
  - **info** \- _string_, type of action: `"cancel"`, `"redirect"`;
  - **message** \- _string_, `"ok"` or `"error"`;
  - **details** \- _object_, info about the request and rule:
    - **rule** \- _object_, the triggered rule;
    - **url** \- _string_, URL of the request;
    - **redirect\_url** \- _string_, where the request was redirected;
    - **description** \- _string_, error description.

### Example

```ts
GM_webRequest([\
    { selector: '*cancel.me/*', action: 'cancel' },\
    { selector: { include: '*', exclude: 'http://exclude.me/*' }, action: { redirect: 'http://new_static.url' } },\
    { selector: { match: '*://match.me/*' }, action: { redirect: { from: '([^:]+)://match.me/(.*)',  to: '$1://redirected.to/$2' } } }\
], function(info, message, details) {
    console.log(info, message, details);
});
```

## GM\_cookie.list(details\[, callback\])

Note: `httpOnly` cookies are supported at the BETA versions of Tampermonkey only for now

Tampermonkey checks if the script has `@include` or `@match` access to given `details.url` arguments!

### Parameters:

- **details** _object_, containing properties of the cookies to retrieve
  - **url** _string?_, representing the URL to retrieve cookies from (defaults to current document URL)
  - **domain** _string?_, representing the domain of the cookies to retrieve
  - **name** _string?_, representing the name of the cookies to retrieve
  - **path** _string?_, representing the path of the cookies to retrieve
  - **partitionKey** v5.2+ _object_?, representing the [partition key](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies#storage_partitioning) of the cookies, use an empty object to retrieve all cookies
    - **topLevelSite** _string_?, representing the top frame site of the cookies
- **callback** _function?_, to be called when the cookies have been retrieved. The function will be passed two arguments:
  - **cookies** _object\[\]_, representing the retrieved cookies
  - **error** _string_, representing an error message if an error occurred, null otherwise.

The cookie objects have the following properties:

- **domain** _string_, representing the domain of the cookie
- **expirationDate** _number?_, the expiration date of the cookie in seconds since the Unix epoch. If not specified, the cookie never expires.
- **firstPartyDomain** _string?_: the first party domain of the cookie.
- **partitionKey** v5.2+ _object_?, containing the partition key of the cookie
  - **topLevelSite** _string_?, representing the top frame site of the cookie
- **hostOnly** _boolean_, indicating whether the cookie is a host-only cookie
- **httpOnly** _boolean_, indicating whether the cookie is an HTTP-only cookie
- **name** _string_, representing the name of the cookie
- **path** _string_, representing the path of the cookie
- **sameSite** _string_, indicating the SameSite attribute of the cookie
- **secure** _boolean_, indicating whether the cookie requires a secure connection
- **session** _boolean_, indicating whether the cookie is a session cookie
- **value** _string_, representing the value of the cookie

### Example usage:

```ts
// Retrieve all cookies with name "mycookie"
GM_cookie.list({ name: "mycookie" }, function(cookies, error) {
  if (!error) {
    console.log(cookies);
  } else {
    console.error(error);
  }
});

// Retrieve all cookies for the current domain
const cookies = await GM.cookie.list()
console.log(cookies);
```

## GM\_cookie.set(details\[, callback\])

Sets a cookie with the given details. Supported properties are defined [here](https://developer.chrome.com/extensions/cookies#method-set).

### Parameters:

- **details**: An object containing the details of the cookie to be set. The object can have the following properties:
  - **url** _string?_, the URL to associate the cookie with. If not specified, the cookie is associated with the current document's URL.
  - **name** _string_, the name of the cookie.
  - **value** _string_, the value of the cookie.
  - **domain** _string?_, the domain of the cookie.
  - **firstPartyDomain** _string?_: the first party domain of the cookie.
  - **partitionKey** v5.2+ _object_?, containing the [partition key of the cookie](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/cookies#storage_partitioning)
    - **topLevelSite** _string_?, representing the top frame site of the cookie
  - **path** _string?_, the path of the cookie.
  - **secure** _boolean?_, whether the cookie should only be sent over HTTPS.
  - **httpOnly** _boolean?_, whether the cookie should be marked as HttpOnly.
  - **expirationDate** _number?_, the expiration date of the cookie in seconds since the Unix epoch. If not specified, the cookie never expires.
- **callback** _function?_, a function to be called when the operation is complete. The function is passed one argument:
  - **error** _string?_, if there was an error setting the cookie, this contains an error message. Otherwise, it is `undefined`.

### Example:

```ts
GM_cookie.set({
  url: 'https://example.com',
  name: 'name',
  value: 'value',
  domain: '.example.com',
  path: '/',
  secure: true,
  httpOnly: true,
  expirationDate: Math.floor(Date.now() / 1000) + (60 * 60 * 24 * 30) // Expires in 30 days
}, function(error) {
  if (error) {
    console.error(error);
  } else {
    console.log('Cookie set successfully.');
  }
});

GM.cookie.set({
  name: 'name',
  value: 'value'
})
.then(() => {
  console.log('Cookie set successfully.');
})
.catch((error) => {
  console.error(error);
});
```

## GM\_cookie.delete(details, callback)

Deletes a cookie.

### Parameters:

The `details` object can have the following properties:

- **url** _string?_, the URL associated with the cookie. If `url` is not specified, the current document's URL will be used.
- **name** _string_, the name of the cookie to delete.
- **firstPartyDomain** _string?_: the first party domain of the cookie to delete.
- **partitionKey** v5.2+ _object_?, representing the partition key of the cookie to delete
  - **topLevelSite** _string_?, representing the top frame site of the cookies

The `callback` function is optional and will be called when the cookie has been deleted or an error has occurred. It takes one argument:

- **error** _string?_, an error message, or `undefined` if the cookie was deleted successfully.

### Example:

```ts
GM_cookie.delete({ name: 'cookie_name' }, function(error) {
    if (error) {
        console.error(error);
    } else {
        console.log('Cookie deleted successfully');
    }
});
```

## GM\_audio.setMute(details, callback?)

Sets the mute state of the current tab.

**Parameters**

- **details** _object_, describing the new mute state of the tab:
  - **isMuted** _boolean_, `true` to mute the tab, `false` to un‑mute it.
- **callback** _(optional)_ _function?_, called when the operation finishes.
  - **error** _(optional)_ _string_, contains an error message if setting the mute state fails, otherwise it is `undefined`.

**Return value**

- _Callback style_: nothing (result is delivered via the callback).
- _Promise style_: returns a `Promise<void>` that resolves on success and rejects with an error string on failure.

**Example (callback)**

```ts
// ==UserScript==
...
// @grant      GM_audio
// ==/UserScript==

GM_audio.setMute({ isMuted: true }, function(err) {
  if (err) console.error('mute failed:', err);
  else console.log('tab muted');
});
```

**Example (Promise)**

```ts
// ==UserScript==
...
// @grant      GM.audio
// ==/UserScript==

await GM.audio.setMute({ isMuted: false });
console.log('tab un‑muted');
```

## GM\_audio.getState(callback)

Retrieves the current audio state of the tab.

**Parameters**

- **callback** _function_, to be called with an object describing the tab’s audio state:
  - **info** _object_, representing the retrieved state
    - **isMuted** _(optional)_ _boolean_, whether the tab is currently muted.
    - **muteReason** _(optional)_ _string_, the reason why the tab was muted, if it is currently muted.
      - `user` – User action (e.g., mute button).
      - `capture` – Tab capture API call.
      - `extension` – Extension call.
    - **isAudible** _(optional)_ _boolean_, whether the tab is currently playing audio.

**Return value**

- _Callback style_: nothing (state delivered via the callback).
- _Promise style_: returns a `Promise` that resolves with the callback’s `info` object on success or rejects on error.

**Example (callback)**

```ts
// ==UserScript==
...
// @grant      GM_audio
// ==/UserScript==

GM_audio.getState(function(state) {
  if (!state) return console.error('failed to read state');
  console.log('muted?', state.isMuted, 'reason:', state.muteReason);
  console.log('audible?', state.isAudible);
});
```

**Example (Promise)**

```ts
// ==UserScript==
...
// @grant      GM.audio
// ==/UserScript==

const state = await GM.audio.getState();
console.log(`muted=${state.isMuted} (reason=${state.muteReason}) audible=${state.isAudible}`);
```

## GM\_audio.addStateChangeListener(listener, callback)

Registers a listener that is called whenever the tab’s mute or audible state changes.

**Parameters**

- **listener** _function_, to be called when state changes. The function will be passed one argument:
  - **info** _object_, representing the retrieved state change
    - **muted** _(optional)_ _string \| false_, mute reason or `false` if not muted.
    - **audible** _(optional)_ _boolean_, whether the tab is currently playing audio.
- **callback** _(optional)_ _function?_, called once the registration attempt is complete. The function will be passed one argument:
  - **error** _(optional)_ _string?_, containing an error message if registration fails, or `undefined` otherwise.

**Return value**

- _Callback style_: nothing (listener registered via callback).
- _Promise style_: returns a `Promise<void>` that resolves when the listener has been successfully registered.

**Example (callback)**

```ts
// ==UserScript==
...
// @grant      GM_audio
// ==/UserScript==

GM_audio.addStateChangeListener(function(e) {
  if ('muted' in e) console.log('muted:', e.muted);
  if ('audible' in e) console.log('audible:', e.audible);
});
```

**Example (Promise)**

```ts
// ==UserScript==
...
// @grant      GM.audio
// ==/UserScript==

await GM.audio.addStateChangeListener(ev => {
  if (ev.muted) console.log('muted by', ev.muted);
});
```

## GM\_audio.removeStateChangeListener(listener, callback)

Unregisters a previously added state‑change listener.

**Parameters**

- **listener** _function_, The exact listener function that was passed to `addStateChangeListener`:
- **callback** _(optional)_ _function?_, called once the listener has been removed

**Return value**

- _Callback style_: nothing.
- _Promise style_: returns a `Promise<void>` that resolves when the listener has been removed.

**Example (callback)**

```ts
// ==UserScript==
...
// @grant      GM_audio
// ==/UserScript==

function onAudio(ev) { console.log(ev); }
GM_audio.addStateChangeListener(onAudio);
...
GM_audio.removeStateChangeListener(onAudio, () => console.log('listener removed'));
```

**Example (Promise)**

```ts
// ==UserScript==
...
// @grant      GM.audio
// ==/UserScript==

await GM.audio.removeStateChangeListener(onAudio);
console.log('listener removed');
```

## window.onurlchange

If a script runs on a single-page application, then it can use `window.onurlchange` to listen for URL changes:

```ts
// ==UserScript==
...
// @grant window.onurlchange
// ==/UserScript==

if (window.onurlchange === null) {
    // feature is supported
    window.addEventListener('urlchange', (info) => ...);
}
```

## window.close

Usually JavaScript is not allowed to close tabs via `window.close`.
Userscripts, however, can do this if the permission is requested via `@grant`.

Note: for security reasons it is not allowed to close the last tab of a window.

```ts
// ==UserScript==
...
// @grant window.close
// ==/UserScript==

if (condition) {
    window.close();
}
```

## window.focus

`window.focus` brings the window to the front, while `unsafeWindow.focus` may fail due to user settings.

```ts
// ==UserScript==
...
// @grant window.focus
// ==/UserScript==

if (condition) {
    window.focus();
}
```

## <><!\[CDATA\[...\]\]></>

CDATA-based way of storing meta data is supported via compatibility option.
Tampermonkey tries to automatically detect whether a script needs this option to be enabled.

```ts
var inline_src = (<><![CDATA[\
    console.log('Hello World!');\
]]></>).toString();

eval(inline_src);
```

# Settings

## Content Script API

Script execution is handled by wrapper code that runs or injects the actual userscripts.
There are various methods and APIs available for this, and the `Content Script API` setting in Tampermonkey determines how and where the wrapper code is executed.

This setting is available in Firefox and Chrome (Manifest V3) versions of the extension.

The following options are available for the Content Script API setting:

- **Content Script**:
Runs the wrapper code as a [content script or via the content script API](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Content_scripts).
This is the default option if not explicitly selected.
Userscripts are retrieved via messaging from the background script, **no** real `document-start` support.
- **UserScripts API**:
Uses the browser's UserScripts API ( [MV3](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/userScripts) \| [MV2](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/userScripts_legacy)) to inject the wrapper code.
  - Chrome: Userscripts are retrieved via messaging from the background script, **no** real `document-start` support.
  - Firefox: The userscript is executed instantly -> `document-start` is supported.
- **UserScripts API Dynamic**:
Uses the browser's UserScripts API ( [MV3](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/userScripts) \| [MV2](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/userScripts_legacy)) to inject both the wrapper code and the userscript code.
The userscript is executed instantly -> `document-start` is supported.

Some known MV3 issues with the Content Script API setting include:

- **Dynamic Mode Limitations**: In Dynamic Mode, `@include` patterns using regular expressions may cause scripts to be injected into every frame.
- **External Resource Updates**: Tampermonkey does not automatically update external `@resource`s.

[▲](https://www.tampermonkey.net/documentation.php# "Scroll to top")

