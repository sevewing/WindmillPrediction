/**
 * Created by Weisi on 01/02/2020
 */
;
(function (window, $, undefined) {
    var gis = (function () {
        var gis = {}, _map

        /**
         * Extend Esri BaseMap Layer
         */
        L.esri.esriMapLayer = function (options) {
            return new L.esri.basemapLayer(options);
        };

        $.extend(gis, {
            /**
             * init map
             * @param divID: id of the 'map' div
             */
            initMap: function (divID) {
                _map = new L.Map('map', {
                    center: [56.2, 10.8],
                    zoom: 8,
                    zoomControl: false,
                    attributionControl: false
                });

                //Gray basemap with labels
                L.esri.esriMapLayer("Gray").addTo(_map);
                L.esri.esriMapLayer("GrayLabels").addTo(_map);

                //Add Zoom control
                L.control.zoom({position: 'topright'}).addTo(_map);

                return this;
            },

            zoomToInit: function () {
                _map.setView([56.2, 10.8], 8);
            }
        });
        return gis;
    })();
    window.sgis = gis;
})
(window, jQuery);
