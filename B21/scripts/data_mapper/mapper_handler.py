import json
import pandas as pd

class MapperHandler(object):
    """
    Class to handle: 
    1. reading in the mapper file
    2. filtering the mapper file to the tags that are needed for this analysis
    3. creating the mapping from object and property name to id (name_to_id)
    4. extracting tags 
    """
    supported_file_types = ['csv', 'xlsx', 'txt']

    def __init__(self, object_name_label, object_id_label, property_name_label, property_id_label) -> None:
        self.object_name_label = object_name_label
        self.object_id_label = object_id_label
        self.property_name_label = property_name_label
        self.property_id_label = property_id_label

    @property
    def labels(self):
        return self.object_name_label, self.object_id_label, self.property_name_label, self.property_id_label

    @classmethod
    def read_mapper(cls, mapper_path, *args, **kwargs):
        """
        Function to read in the the mapper file

        Parameters
        ----------
        mapper_path : str
            path to the mapper file
        args : List[?]
            additional arguments for reading the mapper file
        kwargs : Dict
            additional keyword arguments for reading the mapper file
        
        Returns
        -------
        mapper : pd.DataFrame
            mapper

        Examples
        --------
        import pandas as pd
        from MapperHandler import MapperHandler

        # setup data
        foo = pd.DataFrame({'property_name': ['prop1', 'prop2', 'prop3'], 'object_name': ['obj1', 'obj2', 'obj3'], 'property_id': [1,2,3], 'object_id': [4,5,6]})
        
        mp = MapperHandler('object_name', 'object_id', 'property_name', 'property_id')
        mp.read_mapper(<your/path/here>)
        """
        extension = mapper_path.split('.')[-1]
        assert extension in MapperHandler.supported_file_types, f"Detected of type: {extension}\nSupported types are: {', '.join(MapperHandler.supported_file_types)}"

        # NOTE: think about making the mapper a class attribute ...
        if extension == 'xlsx':
            return pd.read_excel(mapper_path, sheet_name = "Datapoint Mappings")
        return pd.read_csv(mapper_path, *args, **kwargs)

    def filter_mapper(self, mapper, object_ids, property_ids):
        """
        Function to filter the mapper file before going creating a mapping

        Parameters
        ----------
        mapper : pd.DataFrame
            mapper
        object_ids : List[int] | int
            object ids
        property_ids : List[int] | int
            property ids
        
        Returns
        -------
        sub : pd.DataFrame
            filtered mapper

        Examples
        --------
        import pandas as pd
        from MapperHandler import MapperHandler

        # setup data
        foo = pd.DataFrame({'property_name': ['prop1', 'prop2', 'prop3'], 'object_name': ['obj1', 'obj2', 'obj3'], 'property_id': [1,2,3], 'object_id': [4,5,6]})
        
        mp = MapperHandler('object_name', 'object_id', 'property_name', 'property_id')
        sub = mp.filter_mapper(foo, [5,6], [1,2])
        """
        # FIXME: check here if an int is passed ...
        if isinstance(object_ids, int):
            object_ids = [object_ids]
        if isinstance(property_ids, int):
            property_ids = [property_ids]

        # make copy to remove any reference issues
        sub = mapper.loc[( mapper[self.object_id_label].isin(object_ids) ) & (mapper[self.property_id_label].isin(property_ids)), self.labels].copy()
        return sub

    def create_mappings(self, mapper):
        """
        Function to make the mapping between from objectName{sep}PropertyName to ObjectID{sep}PropertyID

        Parameters
        ----------
        mapper : pd.DataFrame
            mapper

        Returns
        -------
        name_to_id : Dict[Tuple[str, str]] -> Tuple[int, int]
            mapping from from (object name, property name) to (object id, property id)
        id_to_name : Dict[Tuple[int, int]] -> Tuple[str, str]
            mapping from (object id, property id) to (object name, property name)

        Examples
        --------
        import pandas as pd
        from MapperHandler import MapperHandler

        # setup data
        foo = pd.DataFrame({'property_name': ['prop1', 'prop2', 'prop3'], 'object_name': ['obj1', 'obj2', 'obj3'], 'property_id': [1,2,3], 'object_id': [4,5,6]})
        
        mp = MapperHandler('object_name', 'object_id', 'property_name', 'property_id')
        # mapper input below is often the filtered mapper ...
        name_to_id, id_to_name = mp.create_mappings(foo)
        """
        # make a list of names and ids to make a dict mapping between the two --- object, property names -> object, property ids
        names = list(zip(mapper[self.object_name_label], mapper[self.property_name_label]))
        ids = list(zip(mapper[self.object_id_label], mapper[self.property_id_label]))

        name_to_id = dict(zip(names, ids))
        id_to_name = dict(zip(ids, names))

        return name_to_id, id_to_name

    def extract_ids(self, mapper):
        """
        Convenience function to extract the ids from the filtered mapping file
        
        Parameters
        ----------
        mapper : pd.DataFrame
            filtered mapper

        Returns
        -------
        object_ids : List[int]
            object ids
        property_ids : List[int]
            property ids    
        
        Examples
        --------
        import pandas as pd
        from MapperHandler import MapperHandler

        # setup data
        foo = pd.DataFrame({'property_name': ['prop1', 'prop2', 'prop3'], 'object_name': ['obj1', 'obj2', 'obj3'], 'property_id': [1,2,3], 'object_id': [4,5,6]})
        
        mp = MapperHandler('object_name', 'object_id', 'property_name', 'property_id')
        # mapper input below is often the filtered mapper ...
        object_ids, property_ids = mp.extract_ids(foo)
        """
        # TODO: check if this function is even needed
        return mapper[self.object_id_label].tolist(), mapper[self.property_id_label].tolist()

    @staticmethod
    def load_mapping(json_path, sep):
        """
        Function to read in a json dictionary (objectname{sep}propertyname: [objectid, propertyid])

        Parameters
        ----------
        json_path : str
            path to the json file 
        sep : str
            separator for the object and property name (key)
        
        Returns
        -------
        name_to_id : Dict[Tuple[str, str]] -> Tuple[int, int]
            mapping from objectname and propertyname to objectid and propertyid
        id_to_name : Dict[Tuple[int, int]] -> Tuple[str, str]
            mapping from objectid and propertyid to objectname and propertyname
        """

        with open(json_path, 'r') as fp:
            compressed = json.load(fp)

        id_to_name = {tuple(v):tuple(k.split(sep)) for k,v in compressed.items()}
        name_to_id = {tuple(k.split(sep)): tuple(v) for k,v in compressed.items()}

        return name_to_id, id_to_name

    @staticmethod
    def save_mapping(mapping, out_path, sep, name_to_id=True):
        """
        Function to save a mapping

        Parameters
        ----------
        mapping : Dict[Tuple[int, int]] -> Tuple[str, str] | Dict[Tuple[str, str]] -> Tuple[int, int]
            id_to_name or name_to_id mapping
        out_path : str
            path to save the mapping
        sep : str
            separator for the object and property name (key)
        name_to_id : bool
            whether the orientation of the dict is name_to_id, default=True --- if false then it assume the dict is id_to_name

        Returns
        -------
        """
        if name_to_id:
            compressed = {sep.join(k): v for k,v in mapping.items()}
        else:
            compressed = {sep.join(v): k for k,v in mapping.items()}

        with open(out_path, 'w') as fp:
            json.dump(compressed, fp, indent=4)

        return

    def read_filter_create_extract(self, mapper_path, object_ids, property_ids, *args, **kwargs):
        """
        High level api to run read_mapper, filter_mapper, create_mappings, extract_tags

        Parameters
        ----------
        mapper_path : str
            path to the mapper file
        object_ids : List[int] | int
            object ids
        property_ids : List[int] | int
            property ids
        args : List[?]
            additional arguments for reading the mapper file
        kwargs : Dict
            additional keyword arguments for reading the mapper file

        Returns
        -------
        mapper : pd.DataFrame
            mapper
        sub : pd.DataFrame
            filtered mapper
        name_to_id : Dict[Tuple[str, str]] -> Tuple[int, int]
            mapping from objectname and propertyname to objectid and propertyid
        id_to_name : Dict[Tuple[int, int]] -> Tuple[str, str]
            mapping from objectid and propertyid to objectname and propertyname
        final_object_ids : List[int]
            object ids  (format that is paired with property ids)
        final_property_ids : List[int]
            property ids (format that is paired with object ids)
        """
        mapper = MapperHandler.read_mapper(mapper_path, *args, **kwargs)
        sub = self.filter_mapper(mapper, object_ids=object_ids, property_ids=property_ids).drop_duplicates()
        name_to_id, id_to_name = self.create_mappings(sub)
        final_obj_ids, final_property_ids = self.extract_ids(sub)

        return mapper, sub, name_to_id, id_to_name, final_obj_ids, final_property_ids
        
