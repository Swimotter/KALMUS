from kalmus.barcodes.Barcode import Barcode


class InstancableBarcode(Barcode):
    def load_dict_value(self, object_dict):
        pass

    def reshape_barcode(self):
        pass

    def get_barcode(self):
        """
        Return the barcode. If not exist reshape the stored computed values first to get the barcode

        :return: Return the barcode
        :rtype: class:`kalmus.barcodes.Barcode.Barcode`
        """
        if self.barcode is None:
            self.reshape_barcode()
        return super()

    def save_as_json(self, filename=None):
        if self.barcode is None:
            self.reshape_barcode()
        super(filename)

    def generate(self, video_path, num_threads):
        pass