import { SearchIcon } from "../utils/icons.jsx";

export default function SearchField({ value, onChange }) {
  return (
    <label className="search-field">
      <SearchIcon />
      <input
        type="text"
        placeholder="搜尋路線或車號"
        aria-label="搜尋路線或車號"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}
